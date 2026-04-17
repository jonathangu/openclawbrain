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
- learned_route vs graph_prior_only (traces): 7 better, 396 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 403/403 (1)
- learned_route vs graph_prior_only (turns): 7 better, 396 tied, 0 worse
- regressions vs graph_prior_only: 0/403 (0)
- regressions vs no_brain floor: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 61/832 required-context phrases vs graph_prior_only 52/832
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- activation precision: explicit learned-route activation precision is 7/403 across 403 observed candidate turns
- activation precision proxy: selection-divergence proxy activation precision is 7/403 against graph_prior_only
- success-adjusted economics: learned_route used 392.285714 estimated prompt tokens, 0.000491 estimated prompt USD, and 10 ms serve-path latency per incremental win vs graph_prior_only 256.714286, 0.000321, and 8
- fail-open: observed 0/403 degraded learned_route turns in this replay lane

## Diagnostic Mode Summary
| mode | traces | diagnostic top-rank | shared top score | mean quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_brain | 403 | 0 | 0 | 0 | 0/403 | 0/832 | 0 | 403 |
| vector_only | 403 | 2 | 403 | 44.119107 | 403/403 | 63/832 | 0 | 403 |
| graph_prior_only | 403 | 394 | 394 | 43.325062 | 403/403 | 52/832 | 0 | 403 |
| learned_route | 403 | 7 | 401 | 43.995037 | 403/403 | 61/832 | 0 | 806 |

## Diagnostic Pairwise Deltas
| pair | trace outcomes (left/right/tied) | turn outcomes (left/right/tied) | mean quality delta | compile delta sum | required-context delta sum | promotion delta sum |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| no_brain - vector_only | 0-403-0 | 0-403-0 | -44.119107 | -403 | -63 | 0 |
| no_brain - graph_prior_only | 0-403-0 | 0-403-0 | -43.325062 | -403 | -52 | 0 |
| no_brain - learned_route | 0-403-0 | 0-403-0 | -43.995037 | -403 | -61 | 0 |
| vector_only - graph_prior_only | 9-0-394 | 9-0-394 | 0.794045 | 0 | 11 | 0 |
| vector_only - learned_route | 2-0-401 | 2-0-401 | 0.124069 | 0 | 2 | 0 |
| graph_prior_only - learned_route | 0-7-396 | 0-7-396 | -0.669975 | 0 | -9 | 0 |

## Artifacts
- summary: `summary.md`
- closeout: `closeout.json`
- index: `index.json`
- summary tables: `summary-tables.json`
- pairwise deltas: `pairwise-deltas.json`
- win-rate matrix: `win-rate-matrix.json`
- worked traces: `worked-traces.md`
- generation report: `generation-report.json`
