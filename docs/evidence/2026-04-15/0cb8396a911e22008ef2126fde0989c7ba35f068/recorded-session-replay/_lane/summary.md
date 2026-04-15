# Recorded Session Replay Proof Lane Closeout

- verdict: **success_and_proven**
- severity: **none**
- why: 20/20 replay proof bundles generated successfully and produced deterministic aggregate outputs.
- requested traces: 20
- successful traces: 20
- failed traces: 0
- note: winner counts below are internal replay diagnostics only.
- source manifest: `canonical-frozen-20` (canonical_recorded_session_trace_set_manifest.v1, 952aff638de8)

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

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 0 | 17 |
| graph_prior_only | 17 | 17 |
| learned_route | 3 | 20 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| tern-recorded-session-proof | tied | better | graph_prior_only | 100 | 545759a1f367 | a7453179760d |
| trace-comparative-replay | tied | better | graph_prior_only | 100 | 3e58e5403b32 | 97413340d5b8 |
| trace-correction-answer-paths-explicit | tied | better | graph_prior_only | 100 | d41ffad1ab6a | 76e39b8d10d2 |
| trace-correction-deeper-proof-story | better | better | learned_route | 100 | 911b4ed41f12 | 485e6b66de6f |
| trace-correction-mode-paths-explicit | tied | better | graph_prior_only | 100 | d887e4564f49 | a9276fe5945f |
| trace-correction-rollout-verdict | tied | better | graph_prior_only | 70 | f92acff724f3 | 99dc27aefe2e |
| trace-direct-answer-proof-bundle-layout | tied | better | graph_prior_only | 88 | 2565d2a61498 | 8118d2f44f24 |
| trace-direct-answer-release-verify | tied | better | graph_prior_only | 100 | bac0e0dc9ddc | 2b79d5f8a8fe |
| trace-direct-answer-reproduce-eval-command | tied | better | graph_prior_only | 100 | a2f5cd523399 | 1d3eef9d904f |
| trace-openclaw-replay-freeze-identity | tied | better | graph_prior_only | 100 | 1bb21b8fb62b | b20e541eacde |
| trace-plan-lane-handoff | tied | better | graph_prior_only | 100 | 6916f60c8bb3 | 64127b61e6d0 |
| trace-plan-proof-artifact-triage | tied | better | graph_prior_only | 100 | 7a2ce481ef19 | 82afa5918c63 |
| trace-plan-regression-workflow | tied | better | graph_prior_only | 88 | 7d7f09e4cb64 | f523ec89a8c0 |
| trace-retrieval-proof-hashes | tied | better | graph_prior_only | 100 | 4f72fc56b238 | 4e9fe403b4a6 |
| trace-retrieval-restart-checklist-lookup | tied | better | graph_prior_only | 100 | 9fe2290407bf | 0fc01587af55 |
| trace-retrieval-routing-prior-doc | tied | better | graph_prior_only | 100 | 620c29751e10 | 750d3084eb45 |
| trace-score-resolution | better | better | learned_route | 100 | 59b200ffa348 | 70e1e890ae54 |
| trace-seed-carry-forward | tied | better | graph_prior_only | 100 | fc5f1064e38f | c149412e84de |
| trace-seed-carry-forward-eval-dedup | better | better | learned_route | 100 | e192687dbafe | bf14b161e80c |
| trace-train-freeze-eval | tied | better | graph_prior_only | 100 | 58ba7b8463ae | 327801c8d40c |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-62397f0ca78c87fcf1911f796713801a4be00053b7c5f2783b07135cdd9e705b |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-ccfbee730ddb226c2c49f4ef760c3bff7d99cde222c4af4689432c61b9800122 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-f3c485cdcfaafeeb8ccc5ec688e2126bf4dcca5b2932d23c338f58fa39de90cc |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-e24e00710191964db795104ce747ecfd07a3e96f13cb0539c8002a3c8793742f |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-3574b30fed5f9a8bd6168b3c9db31f38dc949c61b142cdd3ea75cc2d2f6ef62a |
| worked-traces | worked-traces.md | none | sha256-60ef05ea199f818aebcf2625dfea2dcba63013c9f76d10941a0fecb7bbe1ddcb |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-3aac813c581093d28b27ab03f05d99b502cd22f1f81795af01673a151b02b8ee |
