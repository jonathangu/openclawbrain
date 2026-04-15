# Recorded Session Replay Proof Lane

- requested traces: 20
- successful traces: 20
- failed traces: 0
- mode order: `no_brain`, `vector_only`, `graph_prior_only`, `learned_route`
- note: these lane aggregates are internal deterministic replay diagnostics; use the explainable eval scorecard for public/operator reporting.
- source manifest: `canonical-frozen-20` (canonical_recorded_session_trace_set_manifest.v1, 952aff638de8)
- assumptions: `trace manifest contract=canonical_recorded_session_trace_set_manifest.v1`, `trace manifest setId=canonical-frozen-20`, `No checked-in recorded_session_trace.v1 input in this repo carries provenance strong enough to call it a verified first-party real production trace. This freeze therefore uses replayable equivalents only: 7 sourced directly or normalized from existing repo fixtures and 13 newly frozen equivalents derived from checked-in docs/tests.`

## Explainable Scorecard
- learned_route tie-or-better vs graph_prior_only (traces): 20/20 (1)
- learned_route vs graph_prior_only (traces): 3 better, 17 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 45/45 (1)
- learned_route vs graph_prior_only (turns): 3 better, 42 tied, 0 worse
- regressions vs graph_prior_only: 0/20 (0)
- regressions vs no_brain floor: 0/20 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 71/74 required-context phrases vs graph_prior_only 65/74
- correction absorption: observed 25 feedback-bearing turns (22 non-approval), but replay-lane outputs do not yet measure recurrence after correction
- success-adjusted economics: success-adjusted economics are not computed in replay-lane aggregates; use comparative eval or proof-cron for prompt-cost proxy surfaces
- fail-open: fail-open posture is not modeled in recorded-session replay lane aggregates; use proof-cron health surfaces for degraded-serve reporting

## Diagnostic Mode Summary
| mode | traces | diagnostic top-rank | shared top score | mean quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_brain | 20 | 0 | 0 | 0 | 0/45 | 0/74 | 0 | 0 |
| vector_only | 20 | 0 | 17 | 92.8 | 45/45 | 65/74 | 0 | 0 |
| graph_prior_only | 20 | 17 | 17 | 92.8 | 45/45 | 65/74 | 0 | 0 |
| learned_route | 20 | 3 | 20 | 97.3 | 45/45 | 71/74 | 23 | 0 |

## Diagnostic Pairwise Deltas
| pair | trace outcomes (left/right/tied) | turn outcomes (left/right/tied) | mean quality delta | compile delta sum | required-context delta sum | promotion delta sum |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| no_brain - vector_only | 0-20-0 | 0-45-0 | -92.8 | -45 | -65 | 0 |
| no_brain - graph_prior_only | 0-20-0 | 0-45-0 | -92.8 | -45 | -65 | 0 |
| no_brain - learned_route | 0-20-0 | 0-45-0 | -97.3 | -45 | -71 | -23 |
| vector_only - graph_prior_only | 0-0-20 | 0-0-45 | 0 | 0 | 0 | 0 |
| vector_only - learned_route | 0-3-17 | 0-3-42 | -4.5 | 0 | -6 | -23 |
| graph_prior_only - learned_route | 0-3-17 | 0-3-42 | -4.5 | 0 | -6 | -23 |

## Artifacts
- summary: `summary.md`
- closeout: `closeout.json`
- index: `index.json`
- summary tables: `summary-tables.json`
- pairwise deltas: `pairwise-deltas.json`
- win-rate matrix: `win-rate-matrix.json`
- worked traces: `worked-traces.md`
- generation report: `generation-report.json`
