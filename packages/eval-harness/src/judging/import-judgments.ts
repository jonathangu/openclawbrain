import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

export type Mode = "smoke" | "production";

type JsonObject = Record<string, unknown>;
type Judgment = {
  trace_id: string;
  candidate_id: string;
  overall_score: number;
  judge_id?: string;
  notes?: string;
  synthetic?: boolean;
  raw: JsonObject;
};

type ImportOptions = {
  mode: Mode;
  runId: string;
  runDir: string;
  judgmentsPath: string;
  outputPath: string;
};

const SMOKE_WARNING = "NOT PRODUCT EVIDENCE / SYNTHETIC PIPELINE VALIDATION ONLY";

function parseArgs(argv: string[]): Record<string, string> {
  const parsed: Record<string, string> = {};
  const normalized = argv.filter((arg) => arg !== "--");
  for (let index = 0; index < normalized.length; index += 1) {
    const arg = normalized[index];
    if (!arg.startsWith("--")) throw new Error(`Unexpected positional argument: ${arg}`);
    const key = arg.slice(2);
    const value = normalized[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    parsed[key] = value;
    index += 1;
  }
  return parsed;
}

function parseMode(value: string | undefined): Mode {
  if (value === "smoke" || value === undefined) return "smoke";
  if (value === "production") return "production";
  throw new Error(`--mode must be smoke or production; got ${value}`);
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim() === "") throw new Error(`${label} is required`);
  return value;
}

function scoreFrom(value: unknown): number | undefined {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) return Number(value);
  return undefined;
}

function normalizeEntry(entry: JsonObject, inheritedTraceId?: string): Judgment {
  const traceId = requireString(entry.trace_id ?? entry.traceId ?? inheritedTraceId, "judgment trace_id");
  const candidateId = requireString(entry.candidate_id ?? entry.candidateId, `candidate_id for ${traceId}`);
  const score = scoreFrom(entry.overall_score ?? entry.overallScore ?? entry.score);
  if (score === undefined || !Number.isFinite(score) || score < 1 || score > 5) {
    throw new Error(`judgment ${traceId}/${candidateId} overall_score must be 1..5`);
  }
  return {
    trace_id: traceId,
    candidate_id: candidateId,
    overall_score: score,
    judge_id: typeof entry.judge_id === "string" ? entry.judge_id : typeof entry.judgeId === "string" ? entry.judgeId : undefined,
    notes: typeof entry.notes === "string" ? entry.notes : undefined,
    synthetic: entry.synthetic === true || entry.mode === "smoke" || entry.source === "synthetic",
    raw: entry,
  };
}

function normalizePayload(payload: unknown): Judgment[] {
  if (Array.isArray(payload)) return payload.flatMap(normalizePayload);
  if (payload === null || typeof payload !== "object") throw new Error("judgment payload must be an object or array");
  const object = payload as JsonObject;
  if (Array.isArray(object.judgments)) {
    const traceId = typeof object.trace_id === "string" ? object.trace_id : typeof object.traceId === "string" ? object.traceId : undefined;
    return object.judgments.map((entry) => normalizeEntry({ synthetic: object.synthetic, mode: object.mode, ...(entry as JsonObject) }, traceId));
  }
  if (object.scores && typeof object.scores === "object" && (typeof object.trace_id === "string" || typeof object.traceId === "string")) {
    const traceId = requireString(object.trace_id ?? object.traceId, "judgment trace_id");
    return Object.entries(object.scores as JsonObject).map(([candidateId, scoreValue]) => normalizeEntry({ trace_id: traceId, candidate_id: candidateId, overall_score: scoreValue, synthetic: object.synthetic, mode: object.mode }));
  }
  return [normalizeEntry(object)];
}

async function readJsonMaybeJsonl(filePath: string): Promise<unknown> {
  const text = await readFile(filePath, "utf8");
  try { return JSON.parse(text); }
  catch (error) {
    const rows = text.split(/\r?\n/u).map((line) => line.trim()).filter(Boolean).map((line) => JSON.parse(line));
    if (rows.length > 0) return rows;
    throw error;
  }
}

async function collectJudgments(judgmentsPath: string): Promise<Judgment[]> {
  try {
    const entries = await readdir(judgmentsPath, { withFileTypes: true });
    const payloads = [];
    for (const entry of entries.filter((entry) => entry.isFile() && /\.(json|jsonl)$/u.test(entry.name)).sort((a, b) => a.name.localeCompare(b.name))) {
      payloads.push(await readJsonMaybeJsonl(join(judgmentsPath, entry.name)));
    }
    return payloads.flatMap(normalizePayload);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOTDIR") throw error;
    return normalizePayload(await readJsonMaybeJsonl(judgmentsPath));
  }
}

function key(traceId: string, candidateId: string): string { return `${traceId}\u0000${candidateId}`; }

export async function importJudgments(options: ImportOptions): Promise<{ ok: true; output_path: string; ledger_rows: number; judgment_count: number }> {
  const ledgerPath = join(options.runDir, "ledger-draft.jsonl");
  const mappingPath = join(options.runDir, "blind-judge-packets", "_private-backend-map.json");
  const ledgerRows = (await readFile(ledgerPath, "utf8")).split(/\r?\n/u).map((line) => line.trim()).filter(Boolean).map((line) => JSON.parse(line) as JsonObject);
  const mapping = JSON.parse(await readFile(mappingPath, "utf8")) as Record<string, { trace_id: string; backend_id: string }>;
  const judgments = await collectJudgments(options.judgmentsPath);

  const expected = new Set(Object.entries(mapping).map(([candidateId, item]) => key(item.trace_id, candidateId)));
  const byCandidate = new Map<string, Judgment>();
  for (const judgment of judgments) {
    if (options.mode === "production" && judgment.synthetic) throw new Error(`synthetic judgment ${judgment.trace_id}/${judgment.candidate_id} is not allowed in production`);
    const judgmentKey = key(judgment.trace_id, judgment.candidate_id);
    if (!expected.has(judgmentKey)) throw new Error(`judgment references unknown blind candidate ${judgment.trace_id}/${judgment.candidate_id}`);
    if (byCandidate.has(judgmentKey)) throw new Error(`duplicate judgment for ${judgment.trace_id}/${judgment.candidate_id}`);
    byCandidate.set(judgmentKey, judgment);
  }
  const missing = [...expected].filter((candidate) => !byCandidate.has(candidate));
  if (missing.length > 0) throw new Error(`every blind candidate must have a judge score; missing ${missing.length}`);

  const backendToCandidate = new Map<string, Judgment>();
  for (const [candidateId, item] of Object.entries(mapping)) {
    const judgment = byCandidate.get(key(item.trace_id, candidateId));
    if (!judgment) throw new Error(`missing judgment ${item.trace_id}/${candidateId}`);
    backendToCandidate.set(`${item.trace_id}\u0000${item.backend_id}`, judgment);
  }

  const judgedRows = ledgerRows.map((row) => {
    const traceId = requireString(row.trace_id, "ledger trace_id");
    const backendId = requireString(row.backend_id, "ledger backend_id");
    const slices = Array.isArray(row.slices) ? row.slices.filter((slice): slice is string => typeof slice === "string") : [];
    const selectedProductBackend = selectedBackendForSlices(slices);
    const judgment = backendToCandidate.get(`${traceId}\u0000${backendId}`);
    if (!judgment) throw new Error(`missing judgment for ledger row ${traceId}/${backendId}`);
    return {
      ...row,
      selected_product_backend: selectedProductBackend,
      selected_product_policy: "ocb.slice-policy.v1",
      product_selected: backendId === selectedProductBackend,
      blind_candidate_id: judgment.candidate_id,
      judge_status: "complete",
      judge_score: judgment.overall_score,
      outcome: outcomeForScore(judgment.overall_score),
      utility_delta: utilityForScore(judgment.overall_score),
      judge_id: judgment.judge_id ?? "unknown-judge",
      judge_notes: judgment.notes ?? "",
      judgment: {
        overall_score: judgment.overall_score,
        judge_id: judgment.judge_id,
        notes: judgment.notes,
        synthetic: judgment.synthetic || undefined,
      },
      evidence_warning: options.mode === "smoke" ? SMOKE_WARNING : undefined,
      counts_as_product_evidence: options.mode === "smoke" ? false : row.counts_as_product_evidence,
    };
  });
  await writeFile(options.outputPath, `${judgedRows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  return { ok: true, output_path: options.outputPath, ledger_rows: judgedRows.length, judgment_count: byCandidate.size };
}

function selectedBackendForSlices(slices: readonly string[]): string {
  if (slices.includes("direct-answer")) return "none";
  if (slices.includes("correction-follow-up") || slices.includes("stale-memory-conflict")) return "correction-only";
  if (slices.includes("continuation") || slices.includes("retrieval-heavy") || slices.includes("tool-heavy")) return "full-ocb";
  return "full-ocb";
}

function outcomeForScore(score: number): "loss" | "tie" | "win" {
  if (score >= 4) return "win";
  if (score <= 2) return "loss";
  return "tie";
}

function utilityForScore(score: number): number {
  if (score >= 5) return 2;
  if (score >= 4) return 1;
  if (score <= 1) return -2;
  if (score <= 2) return -1;
  return 0;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const mode = parseMode(args.mode);
  const runId = args["run-id"] ?? "smoke-pr4";
  const runDir = args["run-dir"] ?? join("eval/results", runId);
  const outputPath = args.output ?? join(runDir, mode === "smoke" ? "ledger-judged.synthetic.jsonl" : "ledger-judged.jsonl");
  const judgmentsPath = requireString(args.judgments, "--judgments");
  console.log(JSON.stringify(await importJudgments({ mode, runId, runDir, judgmentsPath, outputPath }), null, 2));
}

if (process.argv[1]?.endsWith("import-judgments.ts")) {
  main().catch((error) => {
    console.error((error as Error).stack ?? (error as Error).message);
    process.exitCode = 1;
  });
}
