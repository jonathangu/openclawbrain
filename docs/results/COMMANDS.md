# Command Surface

This is the canonical V5 command surface contract. Commands must be non-interactive and reproducible.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:
# Results Command Surface

PR4 adds the eval harness command surface used by later scoreboard lanes.

| Canonical command | Status | Notes |
|---|---:|---|
| `pnpm ocb:eval:run` | implemented | Runs all four eval backends against fixture-backed traces and writes `ledger-draft.jsonl`. |
| `pnpm ocb:eval:make-blind-packets` | implemented | Generates label-hidden blind judge packets from an eval run. |

Smoke fixtures in this lane are labeled:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Smoke commands validate pipeline mechanics only. They must not claim product evidence.

## Canonical Commands

| Command | Purpose | Current implementation / required behavior |
|---|---|---|
| `pnpm ocb:results:schema-test` | Validate results schema package tests. | Fails on invalid ledger rows, enum violations, or trusted derived fields. |
| `pnpm ocb:traces:validate` | Validate trace manifest and admitted trace coverage. | `node scripts/traces/validate.mjs --mode smoke`; smoke passes with labeled synthetic traces. |
| `pnpm ocb:traces:validate:smoke` | Explicit smoke trace validation. | `node scripts/traces/validate.mjs --mode smoke`. |
| `pnpm ocb:traces:validate:production` | Explicit production trace gate. | `node scripts/traces/validate.mjs --mode production`; fails closed under 40 admitted real traces or unmet slice minimums. |
| `pnpm ocb:eval:run` | Run the ablation ladder. | `node packages/eval-harness/src/run.ts`; runs `none`, `correction-only`, `correction+heuristics`, and `full-ocb` uniformly. |
| `pnpm ocb:eval:make-blind-packets` | Generate blind judge packets. | `node packages/eval-harness/src/blind-packets.ts`; removes backend labels and randomizes output order per trace. |
| `pnpm ocb:judgments:import` | Import completed judgments. | Validates required judge fields and rejects missing production judgments. |
| `pnpm ocb:ledger:validate` | Validate judged ledger rows. | Recomputes derived values and rejects hand-edited inconsistencies. |
| `pnpm ocb:results:generate` | Generate `/results` artifacts. | Generates results from ledger rows only, including warnings and per-slice tables. |
| `pnpm ocb:decision:generate` | Generate product decision memo. | Applies fixed thresholds and emits exactly one allowed outcome or a declared blocker. |
| `pnpm ocb:e2e:smoke` | Run full smoke pipeline. | Completes Engineering E2E on 4–8 labeled synthetic traces and writes `RUN_STATE.json`. |
| `pnpm ocb:runtime:decide` | Decide one redacted runtime agent turn. | Deterministically emits `fire` or `stay_silent`, captures a candidate-only runtime event, and optionally exports an admission candidate. |
| `pnpm ocb:traces:from-session-logs` | Build production trace set from local OpenClaw session logs. | Reads raw logs locally, emits only redacted real trace metadata, admits 40 traces through `ocb:traces:admit`, and writes `eval/traces/production.jsonl`. |
| `pnpm ocb:judgments:judge-production` | Judge production blind packets. | Runs a deterministic non-synthetic blind rubric over redacted packets without reading the private backend map or raw transcripts. |
| `pnpm ocb:e2e:production` | Run full production evidence pipeline. | Builds/admit session-log traces, validates slice coverage, runs all four backends, generates blind packets, judges/imports, regenerates `/results`, writes `30_DAY_DECISION.md` and `RUN_STATE.json`. |

If exact command names cannot be supported by the repository, equivalent commands must be mapped here before use.

## Required Smoke Outputs

`pnpm ocb:e2e:smoke` must output:

```text
eval/results/<run-id>/ledger-draft.jsonl
eval/results/<run-id>/blind-judge-packets/
eval/results/<run-id>/ledger-judged.synthetic.jsonl
docs/results/index.md
docs/results/summary.json
docs/results/30_DAY_DECISION.blocked.md
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

## Required Production Outputs

`pnpm ocb:e2e:production` must output:

```text
eval/traces/production.manifest.json
eval/traces/production.jsonl
eval/results/<run-id>/ledger-draft.jsonl
eval/results/<run-id>/blind-judge-packets/
eval/judgments/production-session-logs.json
eval/results/<run-id>/ledger-judged.jsonl
docs/results/index.md
docs/results/summary.json
docs/results/30_DAY_DECISION.md
eval/results/<run-id>/RUN_STATE.json
```

Production `RUN_STATE.json` may set both gates true only after 40 real privacy-scrubbed admitted traces, required slice minimums, complete four-backend eval, blind judging, judged ledger import, result regeneration, and threshold application.

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
- memory snapshot ID
- config hash
- cost measurement mode
- trace provenance/admission metadata

## Real trace admission

- `pnpm ocb:traces:admit -- --candidate <redacted-trace.json>` records a candidate without product-evidence admission.
- `pnpm ocb:traces:admit -- --candidate <redacted-trace.json> --admit` admits only if the candidate is `provenance_type=real`, `privacy_scrubbed=true`, `contains_real_user_data=false`, deterministic, and slice-valid.
- `pnpm ocb:traces:admit:test` validates the admission fail-closed behavior.

Admission writes production trace `input.json`/`provenance.json` plus `eval/traces/production.manifest.json`; production validation still fails closed until 40 admitted real privacy-scrubbed traces and slice minimums exist.

## Runtime trace candidate export

- `pnpm ocb:runtime:export-candidate -- --event <redacted-runtime-event.json> --out <trace-candidate.json>` converts a redacted runtime observation into the `ocb:traces:admit` candidate format.
- `pnpm ocb:runtime:export-candidate -- --event <redacted-runtime-event.json> --out <trace-candidate.json> --admit` exports and immediately runs the admission gate.
- `pnpm ocb:runtime:export-candidate:test` validates runtime export, admission handoff, and fail-closed raw/secret guards.

Runtime export requires `privacy_scrubbed=true`, `contains_real_user_data=false`, deterministic reproducibility metadata, preassigned V5 slice, and no raw/unredacted/secret-like fields. It does not itself make Evidence E2E complete; production validation still requires the full admitted trace set and real judging.

## Production trace status

- `pnpm ocb:traces:production-status` reports admitted real privacy-scrubbed product trace counts, per-slice counts, blockers, and `evidence_e2e_complete` from `eval/traces/production.manifest.json`.
- `pnpm ocb:traces:production-status:test` verifies missing/partial manifests report honest blockers.

`pnpm ocb:traces:validate:production` now targets `eval/traces/production.manifest.json` by default so production collection is separated from synthetic smoke traces.

## Runtime event capture

- `pnpm ocb:runtime:capture-event -- --event <redacted-runtime-event.json>` validates and stores a stable runtime event under `eval/runtime-events/`.
- `pnpm ocb:runtime:capture-event:test` verifies candidate-only event capture, manifest writing, export handoff, and raw/secret rejection.

Captured runtime events are private/generated artifacts and are ignored by git. They are candidate-only and do not count as product evidence until exported and admitted.

## Runtime decision interface

- `pnpm ocb:runtime:decide -- --input <redacted-decision-input.json>` deterministically decides `fire` or `stay_silent` for one redacted agent turn and stores the resulting runtime event through `ocb:runtime:capture-event`.
- `pnpm ocb:runtime:decide -- --input <redacted-decision-input.json> --candidate-out <trace-candidate.json>` also runs `ocb:runtime:export-candidate` against the captured event.
- `pnpm ocb:runtime:decide:test` verifies fire, restraint/silence, privacy rejection, malformed input rejection, and candidate export compatibility.

The decision interface is intentionally minimal: it uses explicit redacted inputs, a deterministic threshold over redacted memory candidates, reproducibility metadata, and no external mutating services. Its outputs remain candidate-only until the normal trace admission, production validation, and judging gates pass.
