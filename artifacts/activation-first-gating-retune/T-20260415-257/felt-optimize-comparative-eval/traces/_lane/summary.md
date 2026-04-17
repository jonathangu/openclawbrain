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
- learned_route tie-or-better vs graph_prior_only (traces): 21/25 (0.84)
- learned_route vs graph_prior_only (traces): 0 better, 21 tied, 4 worse
- learned_route tie-or-better vs graph_prior_only (turns): 21/25 (0.84)
- learned_route vs graph_prior_only (turns): 0 better, 21 tied, 4 worse
- regressions vs graph_prior_only: 4/25 (0.16)
- regressions vs no_brain floor: 0/25 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 0/63 required-context phrases vs graph_prior_only 4/63
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- activation precision: explicit learned-route activation precision is 0/25 across 25 observed candidate turns
- activation precision proxy: selection-divergence proxy activation precision is 0/25 against graph_prior_only
- success-adjusted economics: success-adjusted economics are unavailable because learned_route produced no incremental wins vs graph_prior_only in this replay lane
- fail-open: observed 0/25 degraded learned_route turns in this replay lane

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 0 | 24 |
| graph_prior_only | 25 | 25 |
| learned_route | 0 | 21 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002 | tied | better | graph_prior_only | 40 | a0bf5c61cf36 | ffd7ff5c00ad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 14a55e0bae27 | 90357b6a871a |
| live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002 | tied | better | graph_prior_only | 40 | 74da4bf11193 | 7d1ff56748ae |
| live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002 | tied | better | graph_prior_only | 40 | b83bd0659e8f | d27cf9cf2466 |
| live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002 | worse | better | graph_prior_only | 60 | 80605b0b3660 | 721b2af3de7f |
| live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002 | tied | better | graph_prior_only | 40 | 6ebf2871d9e6 | f2c30528e3c3 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | 19e7e4e703e2 | 74514ba9b1db |
| live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002 | worse | better | graph_prior_only | 60 | b32c8b0e3f5a | 087eb386f85b |
| live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003 | tied | better | graph_prior_only | 40 | 728029e43b84 | 1b34a7ee5e26 |
| live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002 | tied | better | graph_prior_only | 40 | 91a4b3ef7b6d | 2a175c6cc542 |
| live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002 | tied | better | graph_prior_only | 40 | c90d06d9bfb2 | 247b6060342d |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 972437446973 | a8de3e1e3dc7 |
| live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003 | tied | better | graph_prior_only | 40 | e27066a7d3ca | ba203543e0ec |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002 | tied | better | graph_prior_only | 40 | ea0f7752081a | 465f8e3c1d82 |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003 | worse | better | graph_prior_only | 70 | b7bc0e1d1f98 | 5436c0623c46 |
| live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007 | worse | better | graph_prior_only | 60 | 26184db7e408 | 201ea2adc769 |
| live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002 | tied | better | graph_prior_only | 40 | cf5a6f709e8f | 323f4d77897d |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002 | tied | better | graph_prior_only | 40 | 044d6bf7d84b | 674ad6707ac1 |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003 | tied | better | graph_prior_only | 40 | e4f04609fe5a | 3aa9f5db4ecb |
| live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002 | tied | better | graph_prior_only | 40 | 5b1b9108a39a | 8a64b0d7c7d7 |
| live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002 | tied | better | graph_prior_only | 40 | 7ced84e1f276 | f09ad8a0f412 |
| live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002 | tied | better | graph_prior_only | 40 | d34f2465b14a | e6b04b5eff64 |
| live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002 | tied | better | graph_prior_only | 40 | 002d6f23f0b3 | 1b7263632613 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | ed852719f944 | 3b21fa36b7f5 |
| live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002 | tied | better | graph_prior_only | 40 | 0e682ab317ae | 51d02ea67c8a |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-99f49e4a2cde915e2414f6cb9df81ec2d38112a6917877a7ec33295540f1db22 |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-fa2b5b44580169dd77e52f6433e8927f67b04e534c40e93b7793cbca7cd975b6 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-dd4afef33eb89a5754d7feb71c1e5563a6f29a4eae26e62438d5225ef0ef8128 |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-cee119086d65ec99162f7115506f846af99370231a4a4d6eff70570a2055d2ca |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ac58ee71f924f0350ae3afe6b71e3d2fb9d9d9695b1c9f41e1faa5d3396f1d63 |
| worked-traces | worked-traces.md | none | sha256-d0799dddb1a24503b1e4b583cd5367f27dcec052bd6e73e9e71b861ff0bd19d4 |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-4941089bcc24160950f9e0bf05af8e16088cb033ee3c1b233c519a3b6448365f |
