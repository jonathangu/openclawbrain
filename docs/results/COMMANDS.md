# Command Surface

This is the canonical V5 command surface contract. Commands must be non-interactive and reproducible.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Smoke commands validate pipeline mechanics only. They must not claim product evidence.

## Canonical Commands

| Command | Purpose | Required behavior |
|---|---|---|
| `pnpm ocb:results:schema-test` | Validate results schema package tests. | Fails on invalid ledger rows, enum violations, or trusted derived fields. |
| `pnpm ocb:traces:validate` | Validate trace manifest and admitted trace coverage. | Smoke passes with labeled synthetic traces; production fails closed under 40 admitted real traces or unmet slice minimums. |
| `pnpm ocb:eval:run` | Run the ablation ladder. | Runs `none`, `correction-only`, `correction+heuristics`, and `full-ocb` uniformly. |
| `pnpm ocb:eval:make-blind-packets` | Generate blind judge packets. | Removes backend labels and randomizes output order per trace. |
| `pnpm ocb:judgments:import` | Import completed judgments. | Validates required judge fields and rejects missing production judgments. |
| `pnpm ocb:ledger:validate` | Validate judged ledger rows. | Recomputes derived values and rejects hand-edited inconsistencies. |
| `pnpm ocb:results:generate` | Generate `/results` artifacts. | Generates results from ledger rows only, including warnings and per-slice tables. |
| `pnpm ocb:decision:generate` | Generate product decision memo. | Applies fixed thresholds and emits exactly one allowed outcome or a declared blocker. |
| `pnpm ocb:e2e:smoke` | Run full smoke pipeline. | Completes Engineering E2E on 4–8 labeled synthetic traces and writes `RUN_STATE.json`. |

If exact command names cannot be supported by the repository, equivalent commands must be mapped here before use.

## Required Smoke Outputs

`pnpm ocb:e2e:smoke` must output:

```text
eval/results/<run-id>/ledger-draft.jsonl
eval/results/<run-id>/blind-judge-packets/
eval/results/<run-id>/ledger-judged.synthetic.jsonl
docs/results/index.md
docs/results/summary.json
docs/results/30_DAY_DECISION.synthetic.md
eval/results/<run-id>/RUN_STATE.json
```

Smoke `RUN_STATE.json` may set:

```json
{
  "engineering_e2e_complete": true,
  "evidence_e2e_complete": false
}
```

## Required Run State

Every run must write:

```text
eval/results/<run-id>/RUN_STATE.json
```

Required schema:

```json
{
  "run_id": "...",
  "mode": "smoke|production",
  "engineering_e2e_complete": false,
  "evidence_e2e_complete": false,
  "trace_count": 0,
  "real_trace_count": 0,
  "synthetic_trace_count": 0,
  "all_slice_minimums_met": false,
  "all_backends_run": false,
  "blind_packets_generated": false,
  "judging_complete": false,
  "ledger_valid": false,
  "results_generated": false,
  "decision_generated": false,
  "judge_disagreement_within_threshold": false,
  "blockers": []
}
```

## Production Fail-Closed Behavior

Production commands must generate blocker artifacts and avoid product claims if any required evidence is missing:

```text
docs/results/BLOCKERS.md
docs/results/NEXT_DATA_NEEDED.md
docs/results/PARTIAL_COMPLETION.md
eval/results/<run-id>/RUN_STATE.json
```

Production must fail closed when:

- fewer than 40 admitted real traces exist
- slice minimums are not met
- provenance metadata is missing
- privacy approval is missing when required
- tool-heavy traces lack fixtures/read-only mode
- reproducibility metadata is missing
- raw cost fields are missing without `cost_measurement_mode`
- memory snapshot metadata is missing
- blind packets are missing
- judge scores are missing
- judge disagreement threshold is exceeded

## Reproducibility Requirements

Commands must avoid interactive prompts and record enough metadata to reproduce a run:

- run ID
- code commit
- eval harness commit
- model ID
- prompt hash
- OCB config hash
- memory snapshot ID and timestamp
- tool fixture version when applicable
