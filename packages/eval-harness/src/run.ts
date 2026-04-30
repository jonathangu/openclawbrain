import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import type { BackendId, BackendResult, EvalBackend } from "./backend-types.ts";
import { correctionHeuristicsBackend } from "./backends/correction-heuristics.ts";
import { correctionOnlyBackend } from "./backends/correction-only.ts";
import { fullOcbBackend } from "./backends/full-ocb.ts";
import { noneBackend } from "./backends/none.ts";
import {
  captureReproducibilityMetadata,
  sha256File,
  type ReproducibilityMetadata,
} from "./reproducibility.ts";
import { createFixtureRuntime, loadToolFixtures } from "./tool-fixtures.ts";
import { loadTraces, SYNTHETIC_EVIDENCE_LABEL, type EvalTrace, type TraceMode } from "./trace.ts";

const ALL_BACKENDS: ReadonlyArray<EvalBackend> = Object.freeze([
  noneBackend,
  correctionOnlyBackend,
  correctionHeuristicsBackend,
  fullOcbBackend,
]);

export interface EvalRunOptions {
  mode: TraceMode;
  runId: string;
  tracesPath: string;
  fixturesPath: string;
  resultsRoot: string;
}

export interface EvalRunSummary {
  run_id: string;
  run_dir: string;
  traces_path: string;
  fixtures_path: string;
  backends: BackendId[];
  trace_count: number;
  output_count: number;
  ledger_path: string;
  run_state_path: string;
}

interface DraftLedgerRow {
  schema_version: "ocb.ledger-draft.v1";
  run_id: string;
  trace_id: string;
  backend_id: BackendId;
  mode: TraceMode;
  provenance_type: string;
  counts_as_product_evidence: boolean;
  evidence_label: string | null;
  admitted: boolean;
  slices: string[];
  backend_output_path: string;
  backend_output_sha256: string;
  judge_status: "not_started";
  judge_score: null;
  judge_rubric_version: null;
  cost_usd: null;
  cost_measurement_mode: "not_measured_local_eval_adapter";
  model_id: null;
  memory_snapshot_id: null;
  reproducibility: ReproducibilityMetadata;
  warnings: string[];
}

export async function runEvalHarness(options: EvalRunOptions): Promise<EvalRunSummary> {
  const runDir = join(options.resultsRoot, options.runId);
  const outputRoot = join(runDir, "outputs");
  await mkdir(outputRoot, { recursive: true });

  const [traces, fixtures, reproducibility] = await Promise.all([
    loadTraces(options.tracesPath, { mode: options.mode }),
    loadToolFixtures(options.fixturesPath),
    captureReproducibilityMetadata({
      runId: options.runId,
      tracePath: options.tracesPath,
      fixturesPath: options.fixturesPath,
      mode: options.mode,
      command: ["node", ...process.argv.slice(1)],
    }),
  ]);
  const tools = createFixtureRuntime(options.fixturesPath, fixtures);
  const ledgerRows: DraftLedgerRow[] = [];

  for (const trace of traces) {
    for (const backend of ALL_BACKENDS) {
      const result = await backend.run(trace, {
        run_id: options.runId,
        reproducibility,
        tools,
      });
      assertUniformResult(trace, backend.id, result);
      const outputPath = join(outputRoot, trace.trace_id, `${safeFilePart(backend.id)}.json`);
      await writeJson(outputPath, result);
      ledgerRows.push({
        schema_version: "ocb.ledger-draft.v1",
        run_id: options.runId,
        trace_id: trace.trace_id,
        backend_id: backend.id,
        mode: trace.mode,
        provenance_type: trace.provenance_type,
        counts_as_product_evidence: trace.counts_as_product_evidence,
        evidence_label: trace.counts_as_product_evidence ? null : SYNTHETIC_EVIDENCE_LABEL,
        admitted: trace.admitted,
        slices: [...trace.slices],
        backend_output_path: outputPath,
        backend_output_sha256: await sha256File(outputPath),
        judge_status: "not_started",
        judge_score: null,
        judge_rubric_version: null,
        cost_usd: null,
        cost_measurement_mode: "not_measured_local_eval_adapter",
        model_id: null,
        memory_snapshot_id: null,
        reproducibility,
        warnings: result.warnings,
      });
    }
  }

  const ledgerPath = join(runDir, "ledger-draft.jsonl");
  await writeFile(ledgerPath, ledgerRows.map((row) => JSON.stringify(row)).join("\n") + "\n");

  const runStatePath = join(runDir, "RUN_STATE.json");
  await writeJson(runStatePath, {
    schema_version: "ocb.eval-run-state.v1",
    run_id: options.runId,
    mode: options.mode,
    engineering_e2e_complete: false,
    evidence_e2e_complete: false,
    all_backends_run: true,
    evidence_blockers:
      options.mode === "production"
        ? ["PR4 does not admit production evidence; later trace/judging gates must complete first."]
        : ["Synthetic smoke data cannot count as product evidence."],
    trace_count: traces.length,
    backend_ids: ALL_BACKENDS.map((backend) => backend.id),
    output_count: ledgerRows.length,
    ledger_draft_path: ledgerPath,
    reproducibility,
  });

  return {
    run_id: options.runId,
    run_dir: runDir,
    traces_path: options.tracesPath,
    fixtures_path: options.fixturesPath,
    backends: ALL_BACKENDS.map((backend) => backend.id),
    trace_count: traces.length,
    output_count: ledgerRows.length,
    ledger_path: ledgerPath,
    run_state_path: runStatePath,
  };
}

function assertUniformResult(
  trace: Readonly<EvalTrace>,
  backendId: BackendId,
  result: BackendResult,
): void {
  if (result.trace_id !== trace.trace_id) {
    throw new Error(`Backend ${backendId} returned mismatched trace_id=${result.trace_id}`);
  }
  if (result.backend_id !== backendId) {
    throw new Error(`Backend ${backendId} returned mismatched backend_id=${result.backend_id}`);
  }
  if (result.external_mutation_allowed !== false) {
    throw new Error(`Backend ${backendId} attempted to allow external mutation`);
  }
}

function defaultRunId(mode: TraceMode): string {
  return `${mode}-${new Date().toISOString().replace(/[:.]/g, "-")}`;
}

function defaultTracesPath(mode: TraceMode): string {
  return mode === "smoke"
    ? "packages/eval-harness/fixtures/smoke-traces.jsonl"
    : "eval/traces/manifest.jsonl";
}

function parseArgs(args: string[]): EvalRunOptions {
  const parsed: Record<string, string> = {};
  const normalizedArgs = args.filter((arg) => arg !== "--");
  for (let index = 0; index < normalizedArgs.length; index += 1) {
    const arg = normalizedArgs[index];
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${arg}`);
    }
    const key = arg.slice(2);
    const value = normalizedArgs[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    parsed[key] = value;
    index += 1;
  }
  const mode = (parsed.mode ?? "smoke") as TraceMode;
  if (mode !== "smoke" && mode !== "production") {
    throw new Error(`Invalid --mode ${mode}`);
  }
  return {
    mode,
    runId: parsed["run-id"] ?? defaultRunId(mode),
    tracesPath: parsed.traces ?? defaultTracesPath(mode),
    fixturesPath: parsed.fixtures ?? "packages/eval-harness/fixtures/tool-fixtures.json",
    resultsRoot: parsed.out ?? "eval/results",
  };
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

function safeFilePart(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}

async function main(): Promise<void> {
  const summary = await runEvalHarness(parseArgs(process.argv.slice(2)));
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

const invokedPath = process.argv[1] ? pathToFileURL(process.argv[1]).href : null;
if (invokedPath === import.meta.url || import.meta.url === pathToFileURL(fileURLToPath(import.meta.url)).href) {
  if (invokedPath === import.meta.url) {
    main().catch((error) => {
      process.stderr.write(`${(error as Error).stack ?? (error as Error).message}\n`);
      process.exitCode = 1;
    });
  }
}
