import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import type { BackendId, BackendResult } from "./backend-types.ts";
import { stableHash } from "./reproducibility.ts";
import { loadTraces, SYNTHETIC_EVIDENCE_LABEL, type EvalTrace } from "./trace.ts";

interface MakeBlindPacketOptions {
  runDir: string;
  tracesPath: string;
  outDir: string;
}

interface LedgerRow {
  run_id: string;
  trace_id: string;
  backend_id: BackendId;
  backend_output_path: string;
  counts_as_product_evidence: boolean;
  evidence_label: string | null;
}

interface BlindCandidate {
  candidate_id: string;
  answer: string;
  intervention: string;
  rationale: string[];
  warnings: string[];
}

interface BlindPacket {
  schema_version: "ocb.blind-judge-packet.v1";
  packet_id: string;
  trace_id: string;
  labels_hidden: true;
  evidence_label: string | null;
  prompt: {
    title: string;
    user_goal: string;
    input_messages: EvalTrace["input_messages"];
    slices: string[];
  };
  candidates: BlindCandidate[];
}

export async function makeBlindPackets(options: MakeBlindPacketOptions): Promise<{
  packet_count: number;
  out_dir: string;
  private_map_path: string;
}> {
  const ledgerRows = await readLedger(join(options.runDir, "ledger-draft.jsonl"));
  const traces = await loadTraces(options.tracesPath, { mode: "smoke" });
  const tracesById = new Map(traces.map((trace) => [trace.trace_id, trace]));
  const rowsByTrace = groupRowsByTrace(ledgerRows);
  const privateMap: Record<string, { trace_id: string; backend_id: BackendId }> = {};
  let packetCount = 0;

  await mkdir(options.outDir, { recursive: true });
  for (const [traceId, rows] of rowsByTrace.entries()) {
    const trace = tracesById.get(traceId);
    if (!trace) {
      throw new Error(`Ledger references missing trace ${traceId}`);
    }
    const candidates: BlindCandidate[] = [];
    for (const row of stableCandidateOrder(rows)) {
      const output = JSON.parse(await readFile(row.backend_output_path, "utf8")) as BackendResult;
      const candidateId = `candidate-${stableHash({ traceId, backendId: row.backend_id }).slice(0, 12)}`;
      privateMap[candidateId] = { trace_id: traceId, backend_id: row.backend_id };
      candidates.push({
        candidate_id: candidateId,
        answer: output.answer,
        intervention: output.intervention,
        rationale: output.rationale,
        warnings: output.warnings,
      });
    }
    const packet: BlindPacket = {
      schema_version: "ocb.blind-judge-packet.v1",
      packet_id: `packet-${stableHash({ traceId }).slice(0, 12)}`,
      trace_id: traceId,
      labels_hidden: true,
      evidence_label: trace.counts_as_product_evidence ? null : SYNTHETIC_EVIDENCE_LABEL,
      prompt: {
        title: trace.title,
        user_goal: trace.user_goal,
        input_messages: trace.input_messages,
        slices: trace.slices,
      },
      candidates,
    };
    await writeFile(join(options.outDir, `${traceId}.json`), `${JSON.stringify(packet, null, 2)}\n`);
    packetCount += 1;
  }

  const privateMapPath = join(options.outDir, "_private-backend-map.json");
  await writeFile(privateMapPath, `${JSON.stringify(privateMap, null, 2)}\n`);
  return { packet_count: packetCount, out_dir: options.outDir, private_map_path: privateMapPath };
}

function groupRowsByTrace(rows: LedgerRow[]): Map<string, LedgerRow[]> {
  const grouped = new Map<string, LedgerRow[]>();
  for (const row of rows) {
    const existingRows = grouped.get(row.trace_id) ?? [];
    existingRows.push(row);
    grouped.set(row.trace_id, existingRows);
  }
  return grouped;
}

function stableCandidateOrder(rows: LedgerRow[]): LedgerRow[] {
  return [...rows].sort((left, right) =>
    stableHash({ traceId: left.trace_id, backendId: left.backend_id }).localeCompare(
      stableHash({ traceId: right.trace_id, backendId: right.backend_id }),
    ),
  );
}

async function readLedger(path: string): Promise<LedgerRow[]> {
  const raw = await readFile(path, "utf8");
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as LedgerRow);
}

function parseArgs(args: string[]): MakeBlindPacketOptions {
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
  const runId = parsed["run-id"] ?? "smoke-pr4";
  const runDir = parsed["run-dir"] ?? join("eval/results", runId);
  return {
    runDir,
    tracesPath: parsed.traces ?? "packages/eval-harness/fixtures/smoke-traces.jsonl",
    outDir: parsed.out ?? join(runDir, "blind-judge-packets"),
  };
}

async function main(): Promise<void> {
  const summary = await makeBlindPackets(parseArgs(process.argv.slice(2)));
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
