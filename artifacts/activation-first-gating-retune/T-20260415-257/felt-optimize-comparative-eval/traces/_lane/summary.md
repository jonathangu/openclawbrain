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
| live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002 | tied | better | graph_prior_only | 40 | 6345756ac46b | ffd7ff5c00ad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | f3ea424b79cb | 90357b6a871a |
| live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002 | tied | better | graph_prior_only | 40 | e7770ad57698 | 5702f983ce3a |
| live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002 | tied | better | graph_prior_only | 40 | bc67671cb886 | 1f7214d8474d |
| live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002 | worse | better | graph_prior_only | 60 | 8db19db504e2 | 30392c983f4f |
| live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002 | tied | better | graph_prior_only | 40 | 67f45bf80d74 | 63f8e44381a3 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | 866b2df0f2f4 | 932f285af25e |
| live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002 | worse | better | graph_prior_only | 60 | a0b60bd866f5 | 4bd82c31616b |
| live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003 | tied | better | graph_prior_only | 40 | 309d84e6f6ea | df7e9d821dc1 |
| live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002 | tied | better | graph_prior_only | 40 | d034ba279bb9 | 329b0f1de1e2 |
| live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002 | tied | better | graph_prior_only | 40 | 3907aa5f0791 | aefc21c45a35 |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | e8c9d3af2f6c | 4ad59a79174a |
| live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003 | tied | better | graph_prior_only | 40 | 8c35ac66aa54 | 59d479d66980 |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002 | tied | better | graph_prior_only | 40 | 983326ba98c9 | 462a481fa6e1 |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003 | worse | better | graph_prior_only | 70 | 92b98fc1e652 | 41477dc28c31 |
| live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007 | worse | better | graph_prior_only | 60 | b15ab7a79207 | 6f9ff1dc6be6 |
| live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002 | tied | better | graph_prior_only | 40 | 9143e3ef423b | 79ee13f9e416 |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002 | tied | better | graph_prior_only | 40 | 6cb3467bdf18 | c74af4cece32 |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003 | tied | better | graph_prior_only | 40 | 47d6fbfe6245 | a5b0b42ef933 |
| live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002 | tied | better | graph_prior_only | 40 | b2e4804646ca | 670d61f94cef |
| live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002 | tied | better | graph_prior_only | 40 | 8aa422c29f02 | ec724b3ad054 |
| live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002 | tied | better | graph_prior_only | 40 | 68c8ca701744 | ff609046ac24 |
| live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002 | tied | better | graph_prior_only | 40 | cf7931001f96 | aa6acafd9a34 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | 1700349d4e0b | fb35bd5499c3 |
| live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002 | tied | better | graph_prior_only | 40 | 13ca61137b3e | 944165d12775 |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-99f49e4a2cde915e2414f6cb9df81ec2d38112a6917877a7ec33295540f1db22 |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-45dcf4124c4f8318444a5df7d3ee6cd5f43072c6a67cbf2aa6b758e50f9d8fdb |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-9ef69da87f3642b6c9370785b07b9db4d4868cb19940b1d597b9acf3e784ba37 |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-44348c52502c0d9839a6873f83820b7424d40ecd694576e474a6ef77298c06f8 |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ac58ee71f924f0350ae3afe6b71e3d2fb9d9d9695b1c9f41e1faa5d3396f1d63 |
| worked-traces | worked-traces.md | none | sha256-d201dfd84f5c598cb584f0666767b9fb9b31eb1c9e567ff2321911d2bd7181a5 |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-4941089bcc24160950f9e0bf05af8e16088cb033ee3c1b233c519a3b6448365f |
