# Recorded Session Replay Proof Lane

- requested traces: 403
- successful traces: 403
- failed traces: 0
- mode order: `no_brain`, `vector_only`, `graph_prior_only`, `learned_route`
- note: these lane aggregates are internal deterministic replay diagnostics; use the explainable eval scorecard for public/operator reporting.
- source manifest: `extracted-semantic-rich-live-535` (frozen_recorded_session_eval_manifest.v1, 26eec14b9bb8)
- assumptions: `Extracted from /Users/guclaw/.openclaw/workspace/task-artifacts/T-20260415-250/semantic-rich-live-535.json`, `Internal local-only live-history replay traces.`, `One-turn traces with prior session messages converted into seed cues.`, `Not approved for public export without a separate redaction pass.`

## Explainable Scorecard
- learned_route tie-or-better vs graph_prior_only (traces): 403/403 (1)
- learned_route vs graph_prior_only (traces): 9 better, 394 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 403/403 (1)
- learned_route vs graph_prior_only (turns): 9 better, 394 tied, 0 worse
- regressions vs graph_prior_only: 0/403 (0)
- regressions vs no_brain floor: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 68/832 required-context phrases vs graph_prior_only 57/832
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- success-adjusted economics: success-adjusted economics are not computed in replay-lane aggregates; use comparative eval or proof-cron for prompt-cost proxy surfaces
- fail-open: fail-open posture is not modeled in recorded-session replay lane aggregates; use proof-cron health surfaces for degraded-serve reporting

## Diagnostic Mode Summary
| mode | traces | diagnostic top-rank | shared top score | mean quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_brain | 403 | 0 | 0 | 0 | 0/403 | 0/832 | 0 | 403 |
| vector_only | 403 | 0 | 403 | 44.367246 | 403/403 | 68/832 | 0 | 403 |
| graph_prior_only | 403 | 394 | 394 | 43.573201 | 403/403 | 57/832 | 0 | 403 |
| learned_route | 403 | 9 | 403 | 44.367246 | 403/403 | 68/832 | 0 | 806 |

## Diagnostic Pairwise Deltas
| pair | trace outcomes (left/right/tied) | turn outcomes (left/right/tied) | mean quality delta | compile delta sum | required-context delta sum | promotion delta sum |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| no_brain - vector_only | 0-403-0 | 0-403-0 | -44.367246 | -403 | -68 | 0 |
| no_brain - graph_prior_only | 0-403-0 | 0-403-0 | -43.573201 | -403 | -57 | 0 |
| no_brain - learned_route | 0-403-0 | 0-403-0 | -44.367246 | -403 | -68 | 0 |
| vector_only - graph_prior_only | 9-0-394 | 9-0-394 | 0.794045 | 0 | 11 | 0 |
| vector_only - learned_route | 0-0-403 | 0-0-403 | 0 | 0 | 0 | 0 |
| graph_prior_only - learned_route | 0-9-394 | 0-9-394 | -0.794045 | 0 | -11 | 0 |

## Artifacts
- summary: `summary.md`
- closeout: `closeout.json`
- index: `index.json`
- summary tables: `summary-tables.json`
- pairwise deltas: `pairwise-deltas.json`
- win-rate matrix: `win-rate-matrix.json`
- worked traces: `worked-traces.md`
- generation report: `generation-report.json`
