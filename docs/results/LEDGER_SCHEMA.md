# Ledger Schema

This is the canonical V5 judged ledger contract.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation rows must display in generated outputs:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Rows with `counts_as_product_evidence=false` must never be aggregated as product proof.

## File Location

Production judged ledger:

```text
eval/results/<run-id>/ledger-judged.jsonl
```

Smoke judged ledger:

```text
eval/results/<run-id>/ledger-judged.synthetic.jsonl
```

Each JSONL row represents one trace/backend/judge record.

## Required Row Shape

```json
{
  "trace_id": "trace-001",
  "source": "telegram|github|session|synthetic|repo-derived|adversarial",
  "provenance_type": "real|synthetic|repo-derived|adversarial",
  "mode": "smoke|production",
  "counts_as_product_evidence": false,
  "privacy_scrubbed": true,
  "slice": "direct-answer|continuation|correction-follow-up|retrieval-heavy|tool-heavy|stale-memory-conflict",
  "priority_class": "primary|secondary",
  "task_type": "...",

  "backend": "none|correction-only|correction+heuristics|full-ocb",
  "memory_fired": true,
  "should_have_fired": true,
  "memory_opportunity_label_source": "pre_run_manifest|labeled_audit",
  "activation_reason": "...",
  "retrieved_memory_ids": ["..."],

  "correctness_delta": 1,
  "usefulness_delta": 1,
  "specificity_delta": 1,
  "raw_quality_delta": 3,
  "normalized_quality_delta": 1,
  "quality_delta": 1,

  "harm_delta": 0,
  "cost_penalty": 0.25,
  "activation_utility": 0.75,

  "abstention_regret": 0,
  "abstention_regret_penalty": 0,
  "net_task_utility": 0.75,

  "false_fire": false,
  "stale_memory_conflict": false,

  "input_tokens": 0,
  "output_tokens": 0,
  "memory_tokens": 0,
  "latency_ms": 0,
  "estimated_cost_usd": 0,
  "cost_measurement_mode": "measured|estimated|bucketed|missing",

  "memory_snapshot_id": "snapshot-...",
  "memory_snapshot_created_at": "2026-04-28T00:00:00Z",
  "ocb_config_hash": "sha256-...",
  "model_id": "...",
  "prompt_hash": "sha256-...",
  "code_commit": "...",
  "eval_harness_commit": "...",

  "judge_mode": "blind_quality|labeled_harm_audit|cost_audit|synthetic_smoke",
  "judge_notes": "...",
  "judge_id": "judge-a",
  "created_at": "2026-04-28T00:00:00Z"
}
```

## Enumerations

### `source`

Allowed values:

- `telegram`
- `github`
- `session`
- `synthetic`
- `repo-derived`
- `adversarial`

### `provenance_type`

Allowed values:

- `real`
- `synthetic`
- `repo-derived`
- `adversarial`

### `mode`

Allowed values:

- `smoke`
- `production`

### `slice`

Allowed values:

- `direct-answer`
- `continuation`
- `correction-follow-up`
- `retrieval-heavy`
- `tool-heavy`
- `stale-memory-conflict`

### `priority_class`

Allowed values:

- `primary`
- `secondary`

Primary slices are fixed as:

- `correction-follow-up`
- `continuation`
- `stale-memory-conflict`

Secondary slices are fixed as:

- `retrieval-heavy`
- `tool-heavy`
- `direct-answer`

### `backend`

Allowed values:

- `none`
- `correction-only`
- `correction+heuristics`
- `full-ocb`

### `cost_measurement_mode`

Allowed values:

- `measured`
- `estimated`
- `bucketed`
- `missing`

Production evidence may not use `missing` unless the generated output records a blocker and excludes the row from product proof.

### `judge_mode`

Allowed values:

- `blind_quality`
- `labeled_harm_audit`
- `cost_audit`
- `synthetic_smoke`

## Derived Field Invariants

The pipeline must derive and validate these fields:

```text
raw_quality_delta = correctness_delta + usefulness_delta + specificity_delta
```

```text
quality_delta = normalized_quality_delta
```

```text
activation_utility = quality_delta - harm_delta - cost_penalty
```

```text
abstention_regret_penalty = abstention_regret * 0.5
```

```text
if memory_fired:
  net_task_utility = activation_utility
elif should_have_fired:
  net_task_utility = -1 * abstention_regret_penalty
else:
  net_task_utility = 0
```

Stored derived values must be checked against recomputation. Hand-edited results must not be trusted as evidence.

## Production Evidence Invariants

A row can count as product evidence only when all are true:

- `mode = "production"`
- `provenance_type = "real"`
- `counts_as_product_evidence = true`
- `privacy_scrubbed = true` or explicit approval is recorded in the trace manifest
- required provenance metadata exists
- reproducibility metadata exists
- required cost fields exist
- memory snapshot metadata exists when a backend can use memory
- the trace was admitted under `docs/results/TRACE_ADMISSION.md`
- judging is complete under `docs/results/JUDGE_PROTOCOL.md`

Smoke rows must always set:

```json
{
  "provenance_type": "synthetic",
  "mode": "smoke",
  "counts_as_product_evidence": false
}
```

## Reproducibility Metadata

Every run must record:

- `memory_snapshot_id`
- `memory_snapshot_created_at`
- `ocb_config_hash`
- `model_id`
- `prompt_hash`
- `code_commit`
- `eval_harness_commit`
- tool fixture version when applicable

## Results Linkage

Every generated `/results` claim must link back to ledger rows. Claims without ledger backing are invalid.
