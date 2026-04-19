# Recorded Session Replay Proof Lane

- requested traces: 25
- successful traces: 25
- failed traces: 0
- mode order: `no_brain`, `vector_only`, `graph_prior_only`, `learned_route`
- note: these lane aggregates are internal deterministic replay diagnostics; use the explainable eval scorecard for public/operator reporting.
- source manifest: `felt_resume_25-eval` (frozen_recorded_session_eval_manifest.v1, 0c68fa167a58)
- assumptions: `accepted manifest contracts: canonical_recorded_session_trace_set_manifest.v1, frozen_recorded_session_eval_manifest.v1`, `manifest trace paths resolve relative to the manifest file location`, `traceHash, when present in the manifest, is checksumJsonPayload(trace-json)`, `scorecard prompt-cost metrics are cheap deterministic proxies derived from selected context chars`, `learned_route is the candidate mode, graph_prior_only is the baseline mode, and no_brain is the floor anchor for the explicit comparative policy`, `when provided, learned_route replay uses the supplied candidate artifact instead of replay-trained route_fn state`, `candidate override replay does not bind the candidate as the served learned-route router, so authoritative broad-live verdicts still require a served-pack bridge`, `this scaffold does not finalize the frozen trace set or widen proof-bundle generation scope`

## Explainable Scorecard
- learned_route tie-or-better vs graph_prior_only (traces): 25/25 (1)
- learned_route vs graph_prior_only (traces): 0 better, 25 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 25/25 (1)
- learned_route vs graph_prior_only (turns): 0 better, 25 tied, 0 worse
- regressions vs graph_prior_only: 0/25 (0)
- regressions vs no_brain floor: 0/25 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 3/63 required-context phrases vs graph_prior_only 3/63
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- activation precision: explicit learned-route activation precision is 0/25 across 25 observed candidate turns
- activation precision proxy: selection-divergence proxy activation precision is 0/25 against graph_prior_only
- success-adjusted economics: success-adjusted economics are unavailable because learned_route produced no incremental wins vs graph_prior_only in this replay lane
- fail-open: observed 0/25 degraded learned_route turns in this replay lane

## Diagnostic Mode Summary
| mode | traces | diagnostic top-rank | shared top score | mean quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_brain | 25 | 0 | 0 | 0 | 0/25 | 0/63 | 0 | 25 |
| vector_only | 25 | 0 | 25 | 42.8 | 25/25 | 3/63 | 0 | 25 |
| graph_prior_only | 25 | 25 | 25 | 42.8 | 25/25 | 3/63 | 0 | 25 |
| learned_route | 25 | 0 | 25 | 42.8 | 25/25 | 3/63 | 0 | 50 |

## Diagnostic Pairwise Deltas
| pair | trace outcomes (left/right/tied) | turn outcomes (left/right/tied) | mean quality delta | compile delta sum | required-context delta sum | promotion delta sum |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| no_brain - vector_only | 0-25-0 | 0-25-0 | -42.8 | -25 | -3 | 0 |
| no_brain - graph_prior_only | 0-25-0 | 0-25-0 | -42.8 | -25 | -3 | 0 |
| no_brain - learned_route | 0-25-0 | 0-25-0 | -42.8 | -25 | -3 | 0 |
| vector_only - graph_prior_only | 0-0-25 | 0-0-25 | 0 | 0 | 0 | 0 |
| vector_only - learned_route | 0-0-25 | 0-0-25 | 0 | 0 | 0 | 0 |
| graph_prior_only - learned_route | 0-0-25 | 0-0-25 | 0 | 0 | 0 | 0 |

## Artifacts
- summary: `summary.md`
- closeout: `closeout.json`
- index: `index.json`
- summary tables: `summary-tables.json`
- pairwise deltas: `pairwise-deltas.json`
- win-rate matrix: `win-rate-matrix.json`
- worked traces: `worked-traces.md`
- generation report: `generation-report.json`
