import { readFile } from "node:fs/promises";

export const SYNTHETIC_EVIDENCE_LABEL =
  "NOT PRODUCT EVIDENCE / SYNTHETIC PIPELINE VALIDATION ONLY";

export type TraceMode = "smoke" | "production";
export type ProvenanceType = "synthetic" | "real" | "repo-derived" | "adversarial";

export interface TraceMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

export interface TraceCorrection {
  summary: string;
  recommended_action: string;
}

export interface TraceToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  fixture_id: string;
  read_only: true;
}

export interface EvalTrace {
  trace_id: string;
  title: string;
  mode: TraceMode;
  provenance_type: ProvenanceType;
  counts_as_product_evidence: boolean;
  privacy_scrubbed: boolean;
  admitted: boolean;
  slices: string[];
  user_goal: string;
  input_messages: TraceMessage[];
  correction?: TraceCorrection;
  tool_calls?: TraceToolCall[];
  expected_behavior?: string;
}

export interface TraceLoadOptions {
  mode: TraceMode;
  allowProductEvidence?: boolean;
}

export async function loadTraces(
  tracesPath: string,
  options: TraceLoadOptions,
): Promise<ReadonlyArray<Readonly<EvalTrace>>> {
  const raw = await readFile(tracesPath, "utf8");
  const traces = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => parseTraceLine(line, index + 1));

  const seen = new Set<string>();
  for (const trace of traces) {
    validateTrace(trace, options);
    if (seen.has(trace.trace_id)) {
      throw new Error(`Duplicate trace_id in ${tracesPath}: ${trace.trace_id}`);
    }
    seen.add(trace.trace_id);
  }

  return traces.map((trace) => deepFreeze(trace));
}

export function validateTrace(trace: EvalTrace, options: TraceLoadOptions): void {
  requireString(trace.trace_id, "trace_id");
  requireString(trace.title, "title");
  requireString(trace.user_goal, "user_goal");
  if (trace.mode !== options.mode) {
    throw new Error(
      `Trace ${trace.trace_id} has mode=${trace.mode}; expected ${options.mode}`,
    );
  }
  if (!Array.isArray(trace.input_messages) || trace.input_messages.length === 0) {
    throw new Error(`Trace ${trace.trace_id} must include input_messages`);
  }
  if (!Array.isArray(trace.slices) || trace.slices.length === 0) {
    throw new Error(`Trace ${trace.trace_id} must include non-empty slices`);
  }
  for (const slice of trace.slices) {
    validateV5Slice(trace.trace_id, slice);
  }
  if (trace.mode === "smoke") {
    assertSyntheticOnly(trace);
  }
  if (trace.mode === "production") {
    assertProductionAdmission(trace, options.allowProductEvidence === true);
  }
  if (trace.tool_calls?.length) {
    validateToolCallsAreFixtureBacked(trace);
  }
}

export function isToolHeavyTrace(trace: EvalTrace): boolean {
  return trace.slices.includes("tool-heavy") || Boolean(trace.tool_calls?.length);
}

function parseTraceLine(line: string, lineNumber: number): EvalTrace {
  try {
    return JSON.parse(line) as EvalTrace;
  } catch (error) {
    throw new Error(`Invalid JSONL trace at line ${lineNumber}: ${(error as Error).message}`);
  }
}

function validateToolCallsAreFixtureBacked(trace: EvalTrace): void {
  for (const toolCall of trace.tool_calls ?? []) {
    requireString(toolCall.id, `tool_call.id for ${trace.trace_id}`);
    requireString(toolCall.name, `tool_call.name for ${trace.trace_id}`);
    requireString(toolCall.fixture_id, `tool_call.fixture_id for ${trace.trace_id}`);
    if (toolCall.read_only !== true) {
      throw new Error(
        `Tool-heavy trace ${trace.trace_id} has non-read-only tool call ${toolCall.id}`,
      );
    }
  }
}

function assertSyntheticOnly(trace: EvalTrace): void {
  if (trace.provenance_type !== "synthetic") {
    throw new Error(`Smoke trace ${trace.trace_id} must use synthetic provenance`);
  }
  if (trace.counts_as_product_evidence !== false) {
    throw new Error(`Smoke trace ${trace.trace_id} cannot count as product evidence`);
  }
}

function assertProductionAdmission(trace: EvalTrace, allowProductEvidence: boolean): void {
  if (!allowProductEvidence) {
    throw new Error(
      "Production evidence mode is intentionally fail-closed in PR4; pass a later admitted manifest gate before enabling it.",
    );
  }
  if (trace.provenance_type !== "real" || trace.admitted !== true) {
    throw new Error(`Production trace ${trace.trace_id} must be admitted real evidence`);
  }
  if (trace.privacy_scrubbed !== true) {
    throw new Error(`Production trace ${trace.trace_id} must be privacy_scrubbed=true`);
  }
}

function validateV5Slice(traceId: string, slice: string): void {
  const allowed = new Set(["direct-answer", "continuation", "correction-follow-up", "retrieval-heavy", "tool-heavy", "stale-memory-conflict"]);
  if (!allowed.has(slice)) {
    throw new Error(`Trace ${traceId} has invalid V5 slice ${slice}`);
  }
}

function requireString(value: unknown, name: string): void {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Missing required string: ${name}`);
  }
}

function deepFreeze<T>(value: T): Readonly<T> {
  if (value && typeof value === "object") {
    Object.freeze(value);
    for (const child of Object.values(value)) {
      deepFreeze(child);
    }
  }
  return value as Readonly<T>;
}
