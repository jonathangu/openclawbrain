import { priorityClassForSlice, type Backend, type TraceSlice } from "../src/rubric.ts";
import { type LedgerRowInput } from "../src/ledger.ts";
let traceCounter = 0;
export function row(overrides: Partial<LedgerRowInput> = {}): LedgerRowInput {
  traceCounter += 1;
  const slice = overrides.slice ?? "correction-follow-up";
  const correctness = overrides.correctness_delta ?? 1;
  const usefulness = overrides.usefulness_delta ?? 1;
  const specificity = overrides.specificity_delta ?? 0;
  const raw = correctness + usefulness + specificity;
  const normalized = raw <= -4 ? -2 : raw <= -2 ? -1 : raw <= 1 ? 0 : raw <= 3 ? 1 : 2;
  const memoryFired = overrides.memory_fired ?? true;
  const shouldHaveFired = overrides.should_have_fired ?? true;
  return {
    trace_id: `trace-${traceCounter}`,
    source: "synthetic",
    provenance_type: "synthetic",
    mode: "smoke",
    counts_as_product_evidence: false,
    privacy_scrubbed: true,
    slice,
    priority_class: priorityClassForSlice(slice),
    task_type: "test-task",
    backend: "full-ocb",
    memory_fired: memoryFired,
    should_have_fired: shouldHaveFired,
    memory_opportunity_label_source: "pre_run_manifest",
    activation_reason: "synthetic test",
    retrieved_memory_ids: [],
    correctness_delta: correctness,
    usefulness_delta: usefulness,
    specificity_delta: specificity,
    raw_quality_delta: raw,
    normalized_quality_delta: normalized,
    quality_delta: normalized,
    harm_delta: 0,
    cost_penalty: 0,
    abstention_regret: 0,
    false_fire: memoryFired && !shouldHaveFired,
    stale_memory_conflict: slice === "stale-memory-conflict",
    input_tokens: 10,
    output_tokens: 10,
    memory_tokens: 0,
    latency_ms: 1,
    estimated_cost_usd: 0,
    cost_measurement_mode: "measured",
    memory_snapshot_id: "snapshot-test",
    memory_snapshot_created_at: "2026-04-28T00:00:00Z",
    ocb_config_hash: "sha256-config",
    model_id: "test-model",
    prompt_hash: "sha256-prompt",
    code_commit: "test-commit",
    eval_harness_commit: "test-harness",
    judge_mode: "synthetic_smoke",
    judge_notes: "synthetic",
    judge_id: "judge-test",
    created_at: "2026-04-28T00:00:00Z",
    ...overrides,
  };
}
export function productionRows(backend: Backend, count: number, opts: Partial<LedgerRowInput> = {}): LedgerRowInput[] {
  const slices: TraceSlice[] = ["correction-follow-up", "continuation", "stale-memory-conflict", "direct-answer", "retrieval-heavy", "tool-heavy"];
  return Array.from({length: count}, (_,i) => {
    const slice = opts.slice ?? slices[i % slices.length];
    return row({ trace_id: `${backend}-${i}`, source: "session", provenance_type: "real", mode: "production", counts_as_product_evidence: true, privacy_scrubbed: true, judge_mode: "blind_quality", judge_id: "judge-a", backend, slice, priority_class: priorityClassForSlice(slice), ...opts });
  });
}
