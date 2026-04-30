export const NON_EVIDENCE_LABEL = "NOT PRODUCT EVIDENCE";
export const SYNTHETIC_PIPELINE_LABEL = "SYNTHETIC PIPELINE VALIDATION ONLY";

export const REQUIRED_BACKENDS = [
  "none",
  "correction-only",
  "correction+heuristics",
  "full-ocb",
] as const;

export const REQUIRED_SLICES = [
  "direct-answer",
  "continuation",
  "correction-follow-up",
  "retrieval-heavy",
  "tool-heavy",
  "stale-memory-conflict",
] as const;

export const LOW_N_TRACE_THRESHOLD = 5;
export const PRODUCT_TRACE_THRESHOLD = 40;

export type EvidenceKind = "real" | "synthetic" | "repo-derived" | "adversarial" | "unknown";

export type LedgerRow = {
  trace_id?: string;
  traceId?: string;
  backend?: string;
  backend_id?: string;
  slice?: string;
  slices?: string[];
  admitted?: boolean;
  provenance_type?: EvidenceKind | string;
  mode?: string;
  counts_as_product_evidence?: boolean;
  privacy_scrubbed?: boolean;
  outcome?: string;
  result?: string;
  winner?: string;
  judge_score?: number | string;
  score?: number | string;
  utility_delta?: number | string;
  net_task_utility?: number | string;
  activation_utility?: number | string;
  cost_usd?: number | string;
  estimated_cost_usd?: number | string;
  failure_modes?: string[];
  harm_flags?: string[];
  negative_result?: boolean;
  [key: string]: unknown;
};

export type NormalizedRow = {
  sourceFile: string;
  lineNumber: number;
  traceId: string;
  backend: string;
  slices: string[];
  admitted: boolean;
  provenanceType: string;
  mode: string;
  countsAsProductEvidence: boolean;
  privacyScrubbed: boolean | null;
  outcome: string;
  judgeScore: number | null;
  utilityDelta: number | null;
  costUsd: number | null;
  failureModes: string[];
  harmFlags: string[];
  negativeResult: boolean;
  raw: LedgerRow;
};

export function normalizeRow(row: LedgerRow, sourceFile: string, lineNumber: number): NormalizedRow {
  const traceId = stringValue(row.trace_id ?? row.traceId) || `missing-trace-id:${sourceFile}:${lineNumber}`;
  const backend = stringValue(row.backend ?? row.backend_id) || "unknown_backend";
  const slices = normalizeSlices(row.slice, row.slices);
  const provenanceType = stringValue(row.provenance_type) || "unknown";
  const mode = stringValue(row.mode) || "unknown";
  const countsAsProductEvidence = row.counts_as_product_evidence === true;
  const utilityDelta = numberValue(row.utility_delta ?? row.net_task_utility ?? row.activation_utility);
  const outcome = normalizeOutcome(stringValue(row.outcome ?? row.result ?? row.winner), utilityDelta, row.negative_result === true);
  const failureModes = stringArray(row.failure_modes);
  const harmFlags = stringArray(row.harm_flags);

  return {
    sourceFile,
    lineNumber,
    traceId,
    backend,
    slices,
    admitted: row.admitted !== false,
    provenanceType,
    mode,
    countsAsProductEvidence,
    privacyScrubbed: typeof row.privacy_scrubbed === "boolean" ? row.privacy_scrubbed : null,
    outcome,
    judgeScore: numberValue(row.judge_score ?? row.score),
    utilityDelta,
    costUsd: numberValue(row.cost_usd ?? row.estimated_cost_usd),
    failureModes,
    harmFlags,
    negativeResult: row.negative_result === true || outcome === "loss" || outcome === "harm" || (utilityDelta !== null && utilityDelta < 0),
    raw: row,
  };
}

export function isSyntheticOrSmoke(row: NormalizedRow): boolean {
  return row.provenanceType === "synthetic" || row.mode === "smoke" || row.countsAsProductEvidence === false;
}

function normalizeSlices(slice: unknown, slices: unknown): string[] {
  const values = Array.isArray(slices) ? slices : slice === undefined ? [] : [slice];
  const clean = values.map((value) => stringValue(value)).filter(Boolean);
  return clean.length > 0 ? [...new Set(clean)] : ["unspecified"];
}

function normalizeOutcome(value: string, utilityDelta: number | null, negativeResult: boolean): string {
  const lowered = value.toLowerCase();
  if (["win", "winner", "helped", "better", "pass"].includes(lowered)) return "win";
  if (["loss", "lost", "worse", "fail", "failed", "regression"].includes(lowered)) return "loss";
  if (["harm", "harmful", "distracted", "unsafe"].includes(lowered)) return "harm";
  if (["tie", "neutral", "same"].includes(lowered)) return "tie";
  if (["abstain", "no_decision", "missing_judgment", "unjudged"].includes(lowered)) return "unjudged";
  if (negativeResult) return "loss";
  if (utilityDelta !== null && utilityDelta > 0) return "win";
  if (utilityDelta !== null && utilityDelta < 0) return "loss";
  return lowered || "unjudged";
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : value === undefined || value === null ? "" : String(value).trim();
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => stringValue(item)).filter(Boolean);
}
