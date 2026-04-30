export const LEDGER_SCHEMA_VERSION = "ocb.results.ledger.v1" as const;
export const SUMMARY_SCHEMA_VERSION = "ocb.results.summary.v1" as const;

export const BACKENDS = ["none", "correction-only", "correction+heuristics", "full-ocb"] as const;
export const TRACE_SLICES = [
  "direct-answer",
  "continuation",
  "correction-follow-up",
  "retrieval-heavy",
  "tool-heavy",
  "stale-memory-conflict",
] as const;
export const PRIMARY_SLICES = ["correction-follow-up", "continuation", "stale-memory-conflict"] as const;
export const SECONDARY_SLICES = ["retrieval-heavy", "tool-heavy", "direct-answer"] as const;
export const SOURCES = ["telegram", "github", "session", "synthetic", "repo-derived", "adversarial"] as const;
export const PROVENANCE_TYPES = ["real", "synthetic", "repo-derived", "adversarial"] as const;
export const RUN_MODES = ["smoke", "production"] as const;
export const PRIORITY_CLASSES = ["primary", "secondary"] as const;
export const MEMORY_OPPORTUNITY_LABEL_SOURCES = ["pre_run_manifest", "labeled_audit"] as const;
export const COST_MEASUREMENT_MODES = ["measured", "estimated", "bucketed", "missing"] as const;
export const JUDGE_MODES = ["blind_quality", "labeled_harm_audit", "cost_audit", "synthetic_smoke"] as const;
export const PRODUCT_OUTCOMES = ["A", "B", "C", "D", "E", "F"] as const;

export type Backend = (typeof BACKENDS)[number];
export type TraceSlice = (typeof TRACE_SLICES)[number];
export type ProvenanceType = (typeof PROVENANCE_TYPES)[number];
export type RunMode = (typeof RUN_MODES)[number];
export type PriorityClass = (typeof PRIORITY_CLASSES)[number];
export type ProductOutcome = (typeof PRODUCT_OUTCOMES)[number];

export function rawQualityDelta(correctness: number, usefulness: number, specificity: number): number {
  return correctness + usefulness + specificity;
}

export function normalizeQualityDelta(raw: number): number {
  if (raw <= -4) return -2;
  if (raw <= -2) return -1;
  if (raw <= 1) return 0;
  if (raw <= 3) return 1;
  return 2;
}

export function deriveActivationUtility(qualityDelta: number, harmDelta: number, costPenalty: number): number {
  return roundUtility(qualityDelta - harmDelta - costPenalty);
}

export function deriveAbstentionRegretPenalty(abstentionRegret: number): number {
  return roundUtility(abstentionRegret * 0.5);
}

export function deriveNetTaskUtility(params: {
  memory_fired: boolean;
  should_have_fired: boolean;
  quality_delta: number;
  harm_delta: number;
  cost_penalty: number;
  abstention_regret: number;
}): number {
  if (params.memory_fired) return deriveActivationUtility(params.quality_delta, params.harm_delta, params.cost_penalty);
  if (params.should_have_fired) return roundUtility(-1 * deriveAbstentionRegretPenalty(params.abstention_regret));
  return 0;
}

export function roundUtility(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

export function priorityClassForSlice(slice: TraceSlice): PriorityClass {
  return (PRIMARY_SLICES as readonly string[]).includes(slice) ? "primary" : "secondary";
}
