# Recorded Session Replay Proof Lane

- requested traces: 20
- successful traces: 20
- failed traces: 0
- mode order: `no_brain`, `vector_only`, `graph_prior_only`, `learned_route`
- source manifest: `canonical-frozen-20` (canonical_recorded_session_trace_set_manifest.v1, 952aff638de8)
- assumptions: `trace manifest contract=canonical_recorded_session_trace_set_manifest.v1`, `trace manifest setId=canonical-frozen-20`, `No checked-in recorded_session_trace.v1 input in this repo carries provenance strong enough to call it a verified first-party real production trace. This freeze therefore uses replayable equivalents only: 7 sourced directly or normalized from existing repo fixtures and 13 newly frozen equivalents derived from checked-in docs/tests.`

## Mode Summary
| mode | traces | ranked winners | shared top score | mean quality | compile ok | phrase hits | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_brain | 20 | 0 | 0 | 0 | 0/45 | 0/74 | 0 | 0 |
| vector_only | 20 | 0 | 17 | 92.8 | 45/45 | 65/74 | 0 | 0 |
| graph_prior_only | 20 | 17 | 17 | 92.8 | 45/45 | 65/74 | 0 | 0 |
| learned_route | 20 | 3 | 20 | 97.3 | 45/45 | 71/74 | 23 | 0 |

## Pairwise Deltas
| pair | trace record | turn record | mean quality delta | compile delta sum | phrase-hit delta sum | promotion delta sum |
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
