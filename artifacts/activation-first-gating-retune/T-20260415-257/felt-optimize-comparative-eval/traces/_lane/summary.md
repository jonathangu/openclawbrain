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
| live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002 | tied | better | graph_prior_only | 40 | 9737eda4c519 | 260f42be064a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 0eda716a52a3 | 1e4888aaf49b |
| live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002 | tied | better | graph_prior_only | 40 | dbbe50e42211 | 20391dea811d |
| live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002 | tied | better | graph_prior_only | 40 | bff36b56aded | 9d889d7f8d60 |
| live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002 | tied | better | graph_prior_only | 60 | c20da61079d8 | e432224057b6 |
| live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002 | tied | better | graph_prior_only | 40 | 9f86ec665069 | 09611af478d7 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | 9b11b1df7f8b | 8ff24b7752a2 |
| live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002 | tied | better | graph_prior_only | 60 | 5d30e068129e | fc033554200f |
| live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003 | tied | better | graph_prior_only | 40 | 10b9eefceeef | 4822fc096b0e |
| live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002 | tied | better | graph_prior_only | 40 | b2034aaa3ad7 | 01d5806276cd |
| live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002 | tied | better | graph_prior_only | 40 | 93c2e94df6fb | aa6af172fe0c |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | af8336e781fa | a6ef475cbe19 |
| live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003 | tied | better | graph_prior_only | 40 | 228f79a854b0 | da4af39e6621 |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002 | tied | better | graph_prior_only | 40 | 3d62430c8a7d | 9537484ab68e |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003 | tied | better | graph_prior_only | 70 | e0575ea2c850 | 1934f98d0df0 |
| live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007 | tied | better | graph_prior_only | 40 | 24649f8c1c05 | 6da7d7607ebf |
| live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002 | tied | better | graph_prior_only | 40 | 3f7242890053 | 8f52fa9c38b1 |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002 | tied | better | graph_prior_only | 40 | b86aa1f8e05c | 1a189266ac78 |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003 | tied | better | graph_prior_only | 40 | c0464d6154f0 | 24650187595a |
| live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002 | tied | better | graph_prior_only | 40 | 6c1a0a97d603 | 9d3eb2e125c1 |
| live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002 | tied | better | graph_prior_only | 40 | cd665d28d645 | d6d2b1fcddc2 |
| live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002 | tied | better | graph_prior_only | 40 | 774afcd5b6e0 | af82d62e9030 |
| live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002 | tied | better | graph_prior_only | 40 | dab047e7eb22 | c0802bea90ac |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | 04a0ba5aeafd | 40532fbbf73f |
| live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002 | tied | better | graph_prior_only | 40 | 19c65db42b68 | d50ee33f7f28 |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-7f77a67c825e638cc392dea67d00696189c9d2b8f91f707320ef2ad9bc15f818 |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-bb133e71cac248bf399ed239ff5c564b491f08d12ef3b74a452e7046ea8e644c |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-0e98edac999ac52e948b108ecc9a3aa9f57dee963b3556c662d1a4406855988c |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-8eac0283cedb7eed13df6787337b20f9d91594434890e207ec2df3ad59d19a3a |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ec21e0a2220a1ca207052ba81098e6500d96cb080f250585025d584fe7bf0ae3 |
| worked-traces | worked-traces.md | none | sha256-3500784f5f0db8a90c530c0c19cb87a8683972a3a6b9313c76d16f1933dcaf3c |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-4941089bcc24160950f9e0bf05af8e16088cb033ee3c1b233c519a3b6448365f |
