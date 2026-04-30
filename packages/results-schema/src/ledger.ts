import { readFileSync } from "node:fs";
import { basename } from "node:path";
import {
  BACKENDS, COST_MEASUREMENT_MODES, JUDGE_MODES, LEDGER_SCHEMA_VERSION,
  MEMORY_OPPORTUNITY_LABEL_SOURCES, PRIORITY_CLASSES, PROVENANCE_TYPES,
  RUN_MODES, SOURCES, TRACE_SLICES,
  deriveAbstentionRegretPenalty, deriveActivationUtility, deriveNetTaskUtility,
  normalizeQualityDelta, priorityClassForSlice, rawQualityDelta,
  type Backend, type PriorityClass, type ProvenanceType, type RunMode, type TraceSlice,
} from "./rubric.ts";

export type LedgerRowInput = {
  trace_id: string;
  source: (typeof SOURCES)[number];
  provenance_type: ProvenanceType;
  mode: RunMode;
  counts_as_product_evidence: boolean;
  privacy_scrubbed: boolean;
  slice: TraceSlice;
  priority_class: PriorityClass;
  task_type: string;
  backend: Backend;
  memory_fired: boolean;
  should_have_fired: boolean;
  memory_opportunity_label_source: (typeof MEMORY_OPPORTUNITY_LABEL_SOURCES)[number];
  activation_reason: string;
  retrieved_memory_ids: string[];
  correctness_delta: number;
  usefulness_delta: number;
  specificity_delta: number;
  raw_quality_delta: number;
  normalized_quality_delta: number;
  quality_delta: number;
  harm_delta: number;
  cost_penalty: number;
  abstention_regret: number;
  false_fire: boolean;
  stale_memory_conflict: boolean;
  input_tokens: number;
  output_tokens: number;
  memory_tokens: number;
  latency_ms: number;
  estimated_cost_usd: number;
  cost_measurement_mode: (typeof COST_MEASUREMENT_MODES)[number];
  memory_snapshot_id: string;
  memory_snapshot_created_at: string;
  ocb_config_hash: string;
  model_id: string;
  prompt_hash: string;
  code_commit: string;
  eval_harness_commit: string;
  judge_mode: (typeof JUDGE_MODES)[number];
  judge_notes: string;
  judge_id: string;
  created_at: string;
};

export type LedgerRow = LedgerRowInput & {
  schema_version: typeof LEDGER_SCHEMA_VERSION;
  activation_utility: number;
  abstention_regret_penalty: number;
  net_task_utility: number;
};

export class LedgerValidationError extends Error {
  readonly issues: string[];
  constructor(issues: string[]) {
    super(`invalid ledger row: ${issues.join("; ")}`);
    this.name = "LedgerValidationError";
    this.issues = issues;
  }
}

const DERIVED_FIELDS = ["activation_utility", "abstention_regret_penalty", "net_task_utility"] as const;

export function parseLedgerRow(value: unknown): LedgerRow {
  const issues = validateLedgerRowInput(value);
  if (issues.length > 0) throw new LedgerValidationError(issues);
  const input = stripSchemaVersion(value as LedgerRowInput & {schema_version?: string});
  return {
    schema_version: LEDGER_SCHEMA_VERSION,
    ...input,
    activation_utility: deriveActivationUtility(input.quality_delta, input.harm_delta, input.cost_penalty),
    abstention_regret_penalty: deriveAbstentionRegretPenalty(input.abstention_regret),
    net_task_utility: deriveNetTaskUtility(input),
  };
}

export function validateLedgerRowInput(value: unknown): string[] {
  const issues: string[] = [];
  if (!isRecord(value)) return ["row must be an object"];
  for (const field of DERIVED_FIELDS) if (field in value) issues.push(`${field} is derived and must not be supplied`);
  if ("schema_version" in value && value.schema_version !== LEDGER_SCHEMA_VERSION) issues.push(`schema_version must be ${LEDGER_SCHEMA_VERSION}`);

  requireNonEmptyString(value, "trace_id", issues);
  requireEnum(value, "source", SOURCES, issues);
  requireEnum(value, "provenance_type", PROVENANCE_TYPES, issues);
  requireEnum(value, "mode", RUN_MODES, issues);
  requireBoolean(value, "counts_as_product_evidence", issues);
  requireBoolean(value, "privacy_scrubbed", issues);
  requireEnum(value, "slice", TRACE_SLICES, issues);
  requireEnum(value, "priority_class", PRIORITY_CLASSES, issues);
  requireNonEmptyString(value, "task_type", issues);
  requireEnum(value, "backend", BACKENDS, issues);
  requireBoolean(value, "memory_fired", issues);
  requireBoolean(value, "should_have_fired", issues);
  requireEnum(value, "memory_opportunity_label_source", MEMORY_OPPORTUNITY_LABEL_SOURCES, issues);
  requireString(value, "activation_reason", issues);
  requireStringArray(value, "retrieved_memory_ids", issues);
  requireInteger(value, "correctness_delta", -2, 2, issues);
  requireInteger(value, "usefulness_delta", -2, 2, issues);
  requireInteger(value, "specificity_delta", -1, 1, issues);
  requireInteger(value, "raw_quality_delta", -5, 5, issues);
  requireInteger(value, "normalized_quality_delta", -2, 2, issues);
  requireInteger(value, "quality_delta", -2, 2, issues);
  requireNumber(value, "harm_delta", 0, 3, issues);
  requireOneOfNumber(value, "cost_penalty", [0, 0.25, 0.5, 1], issues);
  requireInteger(value, "abstention_regret", 0, 3, issues);
  requireBoolean(value, "false_fire", issues);
  requireBoolean(value, "stale_memory_conflict", issues);
  for (const field of ["input_tokens", "output_tokens", "memory_tokens", "latency_ms"] as const) requireInteger(value, field, 0, Number.MAX_SAFE_INTEGER, issues);
  requireNumber(value, "estimated_cost_usd", 0, Number.MAX_SAFE_INTEGER, issues);
  requireEnum(value, "cost_measurement_mode", COST_MEASUREMENT_MODES, issues);
  for (const field of ["memory_snapshot_id", "memory_snapshot_created_at", "ocb_config_hash", "model_id", "prompt_hash", "code_commit", "eval_harness_commit", "judge_notes", "judge_id", "created_at"] as const) requireString(value, field, issues);
  requireEnum(value, "judge_mode", JUDGE_MODES, issues);

  if (isRecord(value)) validateInvariants(value, issues);
  return issues;
}

function validateInvariants(value: Record<string, unknown>, issues: string[]): void {
  if (value.mode === "smoke" && value.counts_as_product_evidence === true) issues.push("smoke rows cannot count as product evidence");
  if (value.provenance_type !== "real" && value.counts_as_product_evidence === true) issues.push("only real provenance may count as product evidence");
  if (value.mode === "production" && value.provenance_type === "real" && value.privacy_scrubbed !== true) issues.push("production real rows require privacy_scrubbed=true");
  if (value.cost_measurement_mode === "missing" && value.cost_penalty !== 0) issues.push("missing cost mode requires cost_penalty=0");

  if (typeof value.correctness_delta === "number" && typeof value.usefulness_delta === "number" && typeof value.specificity_delta === "number") {
    const raw = rawQualityDelta(value.correctness_delta, value.usefulness_delta, value.specificity_delta);
    if (value.raw_quality_delta !== raw) issues.push(`raw_quality_delta must equal correctness+usefulness+specificity (${raw})`);
    const normalized = normalizeQualityDelta(raw);
    if (value.normalized_quality_delta !== normalized) issues.push(`normalized_quality_delta must equal ${normalized}`);
    if (value.quality_delta !== normalized) issues.push("quality_delta must equal normalized_quality_delta");
  }
  if (typeof value.slice === "string" && TRACE_SLICES.includes(value.slice as TraceSlice)) {
    const expected = priorityClassForSlice(value.slice as TraceSlice);
    if (value.priority_class !== expected) issues.push(`priority_class for ${value.slice} must be ${expected}`);
  }
  if (typeof value.memory_fired === "boolean" && typeof value.should_have_fired === "boolean") {
    const expectedFalseFire = value.memory_fired && !value.should_have_fired;
    if (value.false_fire !== expectedFalseFire) issues.push(`false_fire must be ${expectedFalseFire}`);
  }
}

export function parseLedgerJsonl(text: string): LedgerRow[] {
  return text.split(/\r?\n/u).map((line) => line.trim()).filter(Boolean).map((line, index) => {
    try { return parseLedgerRow(JSON.parse(line)); }
    catch (error) {
      if (error instanceof SyntaxError) throw new LedgerValidationError([`line ${index + 1}: invalid JSON`]);
      if (error instanceof LedgerValidationError) throw new LedgerValidationError(error.issues.map((issue) => `line ${index + 1}: ${issue}`));
      throw error;
    }
  });
}

function stripSchemaVersion(row: LedgerRowInput & {schema_version?: string}): LedgerRowInput {
  const { schema_version: _schema, ...rest } = row;
  return rest;
}
function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === "object" && value !== null && !Array.isArray(value); }
function requireString(value: Record<string, unknown>, field: string, issues: string[]) { if (typeof value[field] !== "string") issues.push(`${field} must be a string`); }
function requireNonEmptyString(value: Record<string, unknown>, field: string, issues: string[]) { if (typeof value[field] !== "string" || value[field].trim() === "") issues.push(`${field} must be a non-empty string`); }
function requireBoolean(value: Record<string, unknown>, field: string, issues: string[]) { if (typeof value[field] !== "boolean") issues.push(`${field} must be a boolean`); }
function requireEnum<T extends readonly string[]>(value: Record<string, unknown>, field: string, allowed: T, issues: string[]) { if (!allowed.includes(value[field] as T[number])) issues.push(`${field} must be one of ${allowed.join(", ")}`); }
function requireStringArray(value: Record<string, unknown>, field: string, issues: string[]) { if (!Array.isArray(value[field]) || !(value[field] as unknown[]).every((v) => typeof v === "string")) issues.push(`${field} must be an array of strings`); }
function requireInteger(value: Record<string, unknown>, field: string, min: number, max: number, issues: string[]) { const v=value[field]; if (!Number.isInteger(v)) { issues.push(`${field} must be an integer`); return; } if ((v as number)<min || (v as number)>max) issues.push(`${field} must be between ${min} and ${max}`); }
function requireNumber(value: Record<string, unknown>, field: string, min: number, max: number, issues: string[]) { const v=value[field]; if (typeof v!=="number" || !Number.isFinite(v)) { issues.push(`${field} must be a finite number`); return; } if (v<min || v>max) issues.push(`${field} must be between ${min} and ${max}`); }
function requireOneOfNumber(value: Record<string, unknown>, field: string, allowed: readonly number[], issues: string[]) { const v=value[field]; if (typeof v !== "number" || !allowed.includes(v)) issues.push(`${field} must be one of ${allowed.join(", ")}`); }

function runCli(): void {
  const filePath = process.argv[2];
  if (!filePath) { console.error("usage: pnpm ocb:ledger:validate <ledger.jsonl>"); process.exitCode = 2; return; }
  try { const rows = parseLedgerJsonl(readFileSync(filePath, "utf8")); console.log(JSON.stringify({ ok: true, file: filePath, rows: rows.length }, null, 2)); }
  catch (error) { const message = error instanceof Error ? error.message : String(error); console.error(JSON.stringify({ ok: false, file: filePath, error: message }, null, 2)); process.exitCode = 1; }
}
if (basename(process.argv[1] ?? "") === "ledger.ts") runCli();
