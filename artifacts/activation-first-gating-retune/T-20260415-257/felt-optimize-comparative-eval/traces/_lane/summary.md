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
| live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002 | tied | better | graph_prior_only | 40 | ed459669b7e4 | ffd7ff5c00ad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 5512e8e56e06 | 90357b6a871a |
| live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002 | tied | better | graph_prior_only | 40 | f1970589f510 | 6605c6ff4315 |
| live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002 | tied | better | graph_prior_only | 40 | 465f0a0d1a2f | 8ece7580aa3a |
| live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002 | worse | better | graph_prior_only | 60 | e828d5e91edd | 9135c101a85f |
| live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002 | tied | better | graph_prior_only | 40 | 5b004ae7cb41 | 6c6e7aeac77a |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | 5067582cd321 | 4f95d224642f |
| live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002 | worse | better | graph_prior_only | 60 | 8dd111a37d0a | 5a25387c3e59 |
| live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003 | tied | better | graph_prior_only | 40 | 7c2bd8ba2b98 | 1bc44acd9d00 |
| live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002 | tied | better | graph_prior_only | 40 | b7da4e74a805 | e2873f3457b1 |
| live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002 | tied | better | graph_prior_only | 40 | d67dd35a9bd0 | b3b769fcffca |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 8e7886457a59 | 294b691e2946 |
| live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003 | tied | better | graph_prior_only | 40 | 253b4a6e5999 | c93d913e48ea |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002 | tied | better | graph_prior_only | 40 | a808f9714dca | ab380e3d5ca4 |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003 | worse | better | graph_prior_only | 70 | 67757ba8e897 | 8fd374cc73cf |
| live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007 | worse | better | graph_prior_only | 60 | db5f03278bcd | e60ca388b131 |
| live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002 | tied | better | graph_prior_only | 40 | a534b176d303 | dc3254542597 |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002 | tied | better | graph_prior_only | 40 | 6e4a4950bdfe | 0ab349ddf797 |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003 | tied | better | graph_prior_only | 40 | a0f785607f8f | 4e68a46ff2ca |
| live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002 | tied | better | graph_prior_only | 40 | fbcdc0918870 | 1b532a122d6f |
| live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002 | tied | better | graph_prior_only | 40 | 0c4e8e0ed30e | bbcead4f9194 |
| live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002 | tied | better | graph_prior_only | 40 | e85167015d29 | e9fcbaec11b6 |
| live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002 | tied | better | graph_prior_only | 40 | 0e9575291637 | 226258948da9 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | 483068f32bfb | be1c8309f4f8 |
| live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002 | tied | better | graph_prior_only | 40 | 4317a947da82 | 88ae36f8905d |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-99f49e4a2cde915e2414f6cb9df81ec2d38112a6917877a7ec33295540f1db22 |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-cce843205d0d164cefb4a0c1c7bceaf063a7c64c1bc53460fe64dda18b58af47 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-16544f374f07564d7a4c1fbc1ad1c68f305e207588a642818a541aa39a66c797 |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-7be81517f28b0b02a42669ae23ecfbbb0d1a92f7ad67dec513a89f32db84c5a7 |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ac58ee71f924f0350ae3afe6b71e3d2fb9d9d9695b1c9f41e1faa5d3396f1d63 |
| worked-traces | worked-traces.md | none | sha256-ae69c424bc79035601b45414eeeab50b87157325d1ff0e762bf312b82af23b3f |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-4941089bcc24160950f9e0bf05af8e16088cb033ee3c1b233c519a3b6448365f |
