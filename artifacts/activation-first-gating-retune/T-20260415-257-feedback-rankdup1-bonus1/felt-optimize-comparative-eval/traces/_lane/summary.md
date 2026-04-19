# Recorded Session Replay Proof Lane Closeout

- verdict: **success_and_proven**
- severity: **none**
- why: 25/25 replay proof bundles generated successfully and produced deterministic aggregate outputs.
- requested traces: 25
- successful traces: 25
- failed traces: 0
- note: winner counts below are internal replay diagnostics only.
- source manifest: `felt_resume_25-eval` (frozen_recorded_session_eval_manifest.v1, 0c68fa167a58)

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

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 0 | 25 |
| graph_prior_only | 25 | 25 |
| learned_route | 0 | 25 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002 | tied | better | graph_prior_only | 40 | 22fd33fb866b | 260f42be064a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 7abe806107c5 | 1e4888aaf49b |
| live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002 | tied | better | graph_prior_only | 40 | 12ec6c3ee90b | 07af3e4a2bdb |
| live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002 | tied | better | graph_prior_only | 40 | 3a01556ed560 | 563f1db06c25 |
| live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002 | tied | better | graph_prior_only | 60 | 511208afaf01 | 8d80e55fe49a |
| live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002 | tied | better | graph_prior_only | 40 | b2591d4b0d4d | c2f5b02d79b3 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | aa2877168657 | 8d72cd8f4c26 |
| live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002 | tied | better | graph_prior_only | 60 | fe80d2932976 | 84af5f89eb30 |
| live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003 | tied | better | graph_prior_only | 40 | 96683535b1c4 | d4cf1de190b5 |
| live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002 | tied | better | graph_prior_only | 40 | 5ed7b998d20d | c62161ca5c83 |
| live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002 | tied | better | graph_prior_only | 40 | 078656a9706e | ee330413b5ed |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | a52dbf0eb94b | 59878de16bee |
| live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003 | tied | better | graph_prior_only | 40 | 12a57086e35a | 38db68af0c7b |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002 | tied | better | graph_prior_only | 40 | 4e5c891e03ac | b06e508da25d |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003 | tied | better | graph_prior_only | 70 | 8acb9d000b2f | 319b718a3a98 |
| live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007 | tied | better | graph_prior_only | 40 | 9812b74d79d5 | bad935ead268 |
| live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002 | tied | better | graph_prior_only | 40 | 1d9043da7115 | 8088faca4ae0 |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002 | tied | better | graph_prior_only | 40 | 95d5ec9a538c | 04356b15315a |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003 | tied | better | graph_prior_only | 40 | c22a443ad92d | eeab3ad2f2d2 |
| live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002 | tied | better | graph_prior_only | 40 | 5b4d4f568640 | 6f828a32be6f |
| live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002 | tied | better | graph_prior_only | 40 | dff7b3720286 | 3b1bc220c310 |
| live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002 | tied | better | graph_prior_only | 40 | 74df91dfbf47 | 7019b80c3393 |
| live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002 | tied | better | graph_prior_only | 40 | f400841c2046 | e13ef8d6f2f1 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | 4017822955aa | b76d50197451 |
| live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002 | tied | better | graph_prior_only | 40 | 090abe3d33a1 | 78eb837e4bbd |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-7f77a67c825e638cc392dea67d00696189c9d2b8f91f707320ef2ad9bc15f818 |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-f1346caea0497098d07d972a15ad2ef4b9b52ef2dff011aef211b85f9114b4b2 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-32b9c762ca37e515c05ba695a148bb2b82f6eb1353be0faf66761ae531ee6f26 |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-f02e96a631dca50bf0f9b4d258a6a274acd9148a90d989e2db8ebe02b4fecee5 |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ec21e0a2220a1ca207052ba81098e6500d96cb080f250585025d584fe7bf0ae3 |
| worked-traces | worked-traces.md | none | sha256-91f8eeede9ca9fb3ed75a36906e240387e1c9932fe6aa4b34fac9af906d84e5e |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-e6a81c5315166342cde19584621818d7c207f345ef4e3cab6d2bd1d832a64d6a |
