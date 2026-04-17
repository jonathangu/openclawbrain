# Recorded Session Replay Proof Lane Closeout

- verdict: **success_and_proven**
- severity: **none**
- why: 403/403 replay proof bundles generated successfully and produced deterministic aggregate outputs.
- requested traces: 403
- successful traces: 403
- failed traces: 0
- note: winner counts below are internal replay diagnostics only.
- source manifest: `extracted-semantic-rich-live-535` (frozen_recorded_session_eval_manifest.v1, 26eec14b9bb8)

## Explainable Scorecard
- learned_route tie-or-better vs graph_prior_only (traces): 403/403 (1)
- learned_route vs graph_prior_only (traces): 5 better, 398 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 403/403 (1)
- learned_route vs graph_prior_only (turns): 5 better, 398 tied, 0 worse
- regressions vs graph_prior_only: 0/403 (0)
- regressions vs no_brain floor: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 60/832 required-context phrases vs graph_prior_only 54/832
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- activation precision: explicit learned-route activation precision is 5/403 across 403 observed candidate turns
- activation precision proxy: selection-divergence proxy activation precision is 5/403 against graph_prior_only
- success-adjusted economics: learned_route used 422 estimated prompt tokens, 0.000528 estimated prompt USD, and 10 ms serve-path latency per incremental win vs graph_prior_only 261.6, 0.000327, and 7
- fail-open: observed 0/403 degraded learned_route turns in this replay lane

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 3 | 403 |
| graph_prior_only | 395 | 395 |
| learned_route | 5 | 400 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | tied | better | graph_prior_only | 60 | 36df9ad4feaa | edad5498eb9f |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | tied | better | graph_prior_only | 40 | c518605d4f98 | ca383356bb66 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | tied | better | graph_prior_only | 70 | 831e0714cd74 | a76d4c440d71 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | tied | better | graph_prior_only | 80 | 2a9ad31b5cd3 | c5ec6ed73b8c |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | tied | better | graph_prior_only | 40 | 55164c9195dd | 44a6a6839bf9 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | tied | better | graph_prior_only | 60 | e3943b78adcc | bc741dad2f13 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | tied | better | graph_prior_only | 60 | 37b5912eb7ab | 1b628015914b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | tied | better | graph_prior_only | 60 | 2dbb9f622d01 | 98bd5ce4eb1b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | tied | better | graph_prior_only | 60 | 39d9487e1846 | e1ad04282418 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | tied | better | graph_prior_only | 40 | 39abd75361ca | 6655513668c5 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | tied | better | graph_prior_only | 60 | adbf32559d66 | 78172f57f32b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | tied | better | graph_prior_only | 40 | 5a7e3f615192 | 3c4e4f73413e |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | tied | better | graph_prior_only | 40 | 35c471369315 | e1205e3f9b50 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | tied | better | graph_prior_only | 40 | beabf1abd92b | cbfeca3746c0 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | tied | better | graph_prior_only | 40 | e662cacd9e3d | 8e0e216e2903 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | tied | better | graph_prior_only | 40 | b0f249ebc238 | 099377a41eb5 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | tied | better | graph_prior_only | 60 | 9268d7a1b330 | 0d4e4088e37b |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | tied | better | graph_prior_only | 40 | c01dc9d9b9be | c166eb963051 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | tied | better | graph_prior_only | 40 | d0a4c2943dd1 | 8a4d59b7407f |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | tied | better | graph_prior_only | 40 | 2156677ef0a7 | c895c9bbc628 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | tied | better | graph_prior_only | 40 | 9895ce68b613 | a02dcacdd590 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | tied | better | graph_prior_only | 100 | 9bf1b6b5d5a1 | 261eef95b953 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | tied | better | graph_prior_only | 40 | 1243116df707 | e31a08dd6a99 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | tied | better | graph_prior_only | 40 | 9a1280c5b159 | 2227f8b57bba |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | tied | better | graph_prior_only | 100 | 2e17e0981e7d | e72d779ccf9c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | tied | better | graph_prior_only | 40 | f48dc6cec7ba | 64680a30d92c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | tied | better | graph_prior_only | 40 | a66127ac3209 | bf221acb7c2c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | tied | better | graph_prior_only | 100 | 43ee0a771a10 | b554da25bc09 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | tied | better | graph_prior_only | 40 | a1cdc55199ad | 148fc45e87b7 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | tied | better | graph_prior_only | 40 | 8e5fc5f5eb45 | d5ac0890278e |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | tied | better | graph_prior_only | 40 | 655fab96f818 | 40c13c90116f |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | tied | better | graph_prior_only | 40 | 25a857f4cba6 | 8e1d06404fd1 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | tied | better | graph_prior_only | 60 | 5d0d60bae445 | cf1295f73c8f |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 6a20fd93b610 | 06ed82bc4982 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | tied | better | graph_prior_only | 40 | 78b4eefce1aa | a0a3152dba6a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | tied | better | graph_prior_only | 40 | 0293799eac59 | 9c6eb9a9f323 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | tied | better | graph_prior_only | 40 | 977b60e553fa | 8fcebf98d5c2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | tied | better | graph_prior_only | 40 | e370f5c16d6d | 5080f4454907 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | tied | better | graph_prior_only | 40 | c20466ba4a83 | 3ffe2bc074b6 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | tied | better | graph_prior_only | 40 | 62941cff844a | b4e8ca0398f2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | tied | better | graph_prior_only | 40 | b8aaf8efb5ba | 84cc968bbaad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | tied | better | graph_prior_only | 40 | 56dcf8262ef1 | 9eed19502b06 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | tied | better | graph_prior_only | 40 | c2050de6e358 | 3869501d75bc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | tied | better | graph_prior_only | 40 | 4375a2e7ffaa | 12b2d027e2d3 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | tied | better | graph_prior_only | 40 | 77dc925d10da | ca388d0aec96 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | tied | better | graph_prior_only | 40 | 514ea53e8f71 | 522a1b097477 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | tied | better | graph_prior_only | 40 | 393887aee2d6 | af07c2ecc786 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | tied | better | graph_prior_only | 40 | ffa9586d52b0 | 08bb1d565398 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | tied | better | graph_prior_only | 40 | d66975b159e1 | e519a7a7b811 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | tied | better | graph_prior_only | 40 | 96813a9de13a | 4e9194910279 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | tied | better | graph_prior_only | 60 | 233d83a1fae7 | 96b705fa8599 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | tied | better | graph_prior_only | 40 | b279a3a29d47 | 0ae86cff5aa4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | tied | better | graph_prior_only | 40 | 37e7bdc1109a | e89d13e27085 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | tied | better | graph_prior_only | 60 | 56a315707bd5 | 6c67f4e16b7e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | tied | better | graph_prior_only | 40 | 394d7440c9a8 | 36e47d3ece62 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | tied | better | graph_prior_only | 40 | bbec562d8132 | 220183859626 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | tied | better | graph_prior_only | 40 | 03e80e4c8a40 | ec516ad1fa28 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | tied | better | graph_prior_only | 40 | 0d9315fab127 | a06ea7ba51bb |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | tied | better | graph_prior_only | 40 | 8e033dfddfdc | 59837c234fe0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | tied | better | graph_prior_only | 40 | 63a7db76c255 | bcb3ae37393b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | tied | better | graph_prior_only | 40 | 18b2f1635588 | 86a5a3182c34 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | tied | better | graph_prior_only | 40 | b26197f99c9f | 4c3ee149e11c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | tied | better | graph_prior_only | 40 | 6f6c9c7ed2db | 3907dcaa7402 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | tied | better | graph_prior_only | 40 | 3fc0a6cd7f64 | 72f88b4bdad9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | tied | better | graph_prior_only | 40 | b587c2fa22e2 | 20ac59c08652 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | tied | better | graph_prior_only | 40 | 3942fdd06afb | 88d5399f0775 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | tied | better | graph_prior_only | 40 | 0741fa0322fa | fa3c6152ddf4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | tied | better | graph_prior_only | 40 | a23ed0e46c34 | dbef0109aed8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | tied | better | graph_prior_only | 60 | 25d7c0c7fee8 | 1d8b282e7675 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | tied | better | graph_prior_only | 40 | 28bf6da2bbbc | 6424e6ea8e88 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | tied | better | graph_prior_only | 40 | ac8ca01fd91e | 2c4f4ddd0b09 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | tied | better | graph_prior_only | 40 | e665df2b9a10 | 6c8e0ad784d1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | tied | better | graph_prior_only | 40 | 34c89e78b445 | 04c68c703324 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | tied | better | graph_prior_only | 40 | 47bf5295dfae | b462154e801b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | tied | better | graph_prior_only | 40 | e4f07b609aba | 39f491fc9842 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | tied | better | graph_prior_only | 40 | 6b6036a8fe3c | 396786ed78ce |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | tied | better | graph_prior_only | 40 | b62dfb69c21e | 9debe3ac15b1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | tied | better | graph_prior_only | 60 | 6274baf9e061 | d2a7bc880b86 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | tied | better | graph_prior_only | 40 | f032b5d4cfa5 | 25da4ce903b5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | tied | better | graph_prior_only | 40 | 14d73b55c702 | 780e67901873 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | tied | better | graph_prior_only | 40 | 1a8b03b9083c | ac12b76638a0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | tied | better | graph_prior_only | 40 | 545ed7cce5a6 | 29c6807faf7b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | tied | better | graph_prior_only | 40 | d5880b7125bf | 4c4099275c01 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | tied | better | graph_prior_only | 40 | 623d5b15c7ab | 82a8ccc6d974 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | tied | better | graph_prior_only | 40 | 0a16adefdf53 | 6a2ed0ddc209 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | tied | better | graph_prior_only | 40 | 25286b6c798d | 5f97fde9f9bd |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | tied | better | graph_prior_only | 40 | f04ee6fc5212 | b7019561acc9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | tied | better | graph_prior_only | 40 | c38a2e376f13 | e732c17d1d88 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | tied | better | graph_prior_only | 40 | 938ed0834e86 | 269625b62b50 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | tied | better | graph_prior_only | 40 | ca26b6d1d68b | 7b22f88bb275 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | tied | better | graph_prior_only | 40 | 578c16586192 | 3d174682d401 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | tied | better | graph_prior_only | 40 | e2fb0e9942e5 | d6ee8668a1e7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | tied | better | graph_prior_only | 40 | ce36384be298 | 2f8e1c96c32b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | tied | better | graph_prior_only | 40 | f7a97d8b26b2 | 2e40cf56fac7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | tied | better | graph_prior_only | 70 | 938ca55990e4 | 4b6547a46d43 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | tied | better | graph_prior_only | 40 | a3f8e6f5b3aa | cf3b91722193 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | tied | better | graph_prior_only | 40 | ba2b3a258b12 | 174c81692022 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | tied | better | graph_prior_only | 40 | 55fe24e3632e | 553e91abb22f |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | tied | better | graph_prior_only | 40 | 1e846f6e6175 | 510aa236171b |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | tied | better | graph_prior_only | 40 | 666417039078 | 9407ae1d6428 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | tied | better | graph_prior_only | 40 | 2b2ac9c7c895 | 9aef30dae145 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | tied | better | graph_prior_only | 40 | bb857568b25d | 356a5a83fa9a |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | tied | better | graph_prior_only | 40 | d6da640fc0bc | 1dae0e21ce2c |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | tied | better | graph_prior_only | 40 | 039d223d188f | 81b0529fc6c4 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | tied | better | graph_prior_only | 40 | 8a3cdb45b19a | f45bcf896489 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | tied | better | graph_prior_only | 40 | 08544ba5b13c | bbe24d75b86e |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | tied | better | graph_prior_only | 40 | 1bfec5b44df9 | b575f97900c5 |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | tied | better | graph_prior_only | 40 | b95d1bd08f90 | 81349507f4b0 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | tied | better | graph_prior_only | 40 | 16e32a84f824 | afc548d1ff2b |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | tied | better | graph_prior_only | 40 | 489785e5d44e | e48891cd17c3 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | tied | better | graph_prior_only | 60 | 35e528ae8fe9 | 587e322a84ba |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | tied | better | graph_prior_only | 40 | 11c1b99b8275 | 40aff0ead9dd |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | tied | better | graph_prior_only | 40 | 3dc0cb4cad07 | 4c1bc849bef2 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | tied | better | graph_prior_only | 40 | d797228a71b0 | ea65c47ca021 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | tied | better | graph_prior_only | 100 | 2a482bd811db | 169693c1b9b5 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | tied | better | graph_prior_only | 40 | a30c4b2149e8 | bf56aac9a861 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | tied | better | graph_prior_only | 40 | 00937f5d9890 | 086396ae851b |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | tied | better | graph_prior_only | 40 | 920765d40650 | f085a7f23593 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | tied | better | graph_prior_only | 70 | 5ff48f36efc3 | 84d74b641101 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | tied | better | graph_prior_only | 40 | d7a97bf0539f | 1ecee8c91b98 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | tied | better | graph_prior_only | 40 | f2e6d717fdf0 | 1e0768b14153 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | tied | better | graph_prior_only | 40 | 60ad19f844e6 | 9f862e6b1532 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | tied | better | graph_prior_only | 40 | 092b380e4caa | cbc1f7939e59 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | tied | better | graph_prior_only | 40 | ebd435c0786e | af7d1c391880 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | tied | better | graph_prior_only | 40 | 80698829f783 | 88c54885e650 |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | tied | better | graph_prior_only | 40 | 45f9a26dcc3f | db6f2e886298 |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | tied | better | graph_prior_only | 100 | 3f4491754f15 | bbad194ad8d3 |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | tied | better | graph_prior_only | 40 | d8b0c74121af | f032fdabe0cd |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | tied | better | graph_prior_only | 40 | 24f540f0a654 | 72ad82775753 |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | tied | better | graph_prior_only | 40 | e35505e7a718 | 5baa670987a9 |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | tied | better | graph_prior_only | 80 | bd8c498a2790 | dd270af6fd10 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | dc0bdb272407 | fa704b0dc067 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | tied | better | graph_prior_only | 40 | d7f8edcda46b | 748d1584556e |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | tied | better | vector_only | 60 | 66d7be2da1ff | e47a1c5cb16b |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | tied | better | graph_prior_only | 40 | 351438507664 | c73b24ed136b |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | tied | better | graph_prior_only | 40 | b4f9fb219bff | c9896fc34949 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | tied | better | graph_prior_only | 40 | a86151e61213 | 0102ec7c3b25 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | tied | better | graph_prior_only | 40 | c4d20f1a9a1a | 024c5a1e6dca |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | tied | better | graph_prior_only | 40 | a09dafb532fe | 2d87166b34df |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | tied | better | graph_prior_only | 40 | 8747f0dc45c3 | 77650081c0c8 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | tied | better | graph_prior_only | 40 | 0164ba5569d1 | 2ebb022b0129 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | tied | better | graph_prior_only | 40 | a32b5be43f5c | 0c04f3352bda |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | tied | better | graph_prior_only | 40 | c2b8eb90ddd3 | 022c5d5ec809 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | tied | better | graph_prior_only | 40 | ea11286700c0 | c0d97bddafdc |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | tied | better | graph_prior_only | 40 | 6402bbecf530 | 862ecf61f28c |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | tied | better | graph_prior_only | 40 | 1580a3bf3684 | a34888210d50 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | tied | better | graph_prior_only | 40 | 94c0da2073c0 | 1e37df0fe8d2 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | tied | better | graph_prior_only | 40 | 1f97ee52633e | 293e87c0c409 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | tied | better | graph_prior_only | 40 | e434cca32d6b | d0ba1609573f |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 61345a1a3b9b | 0647f8982657 |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | better | better | learned_route | 80 | eb2688ce6daa | 33ed476d364c |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | tied | better | graph_prior_only | 60 | 85553128e7a7 | 104877178876 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | tied | better | graph_prior_only | 40 | b69e0c2a8fa9 | 74792f1afbdd |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | tied | better | graph_prior_only | 40 | d836f8604a0e | 0f141a6a533b |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | tied | better | graph_prior_only | 40 | ef4122cb1ab3 | 2c235197497e |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | tied | better | graph_prior_only | 40 | 94d18ed6cc4d | cb94e3c51211 |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | tied | better | graph_prior_only | 40 | c9604b99739b | 9b41f1670fdb |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | tied | better | graph_prior_only | 40 | 59bf5a281558 | c83a720544b0 |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | tied | better | graph_prior_only | 60 | a8279bb36f35 | 20b3a3aad173 |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | tied | better | graph_prior_only | 40 | 9bf11253e1fb | d9d7065c486b |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | tied | better | graph_prior_only | 100 | ebf4ee5930b2 | e816ad8952b9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | tied | better | vector_only | 70 | f90c814be749 | 762d117cee37 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | tied | better | graph_prior_only | 40 | f97860392e7f | 8a4a60cb7d2a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | tied | better | graph_prior_only | 40 | 4d3a3e6e37a1 | cb41aee77f92 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | tied | better | graph_prior_only | 40 | 572613ae0dc0 | 235bdf8a120f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | tied | better | graph_prior_only | 40 | 77e8f6a72d5b | 2cf0e1da0d11 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | tied | better | graph_prior_only | 40 | 0283c45c88ad | 5a3566e0092f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | tied | better | graph_prior_only | 40 | 95342566fbe6 | ef73c543e62b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | tied | better | graph_prior_only | 40 | 07a184282a1c | 7aa8c71275d4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | tied | better | graph_prior_only | 40 | f816598329a7 | 27ff24a4df09 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | tied | better | graph_prior_only | 40 | 87572e132a65 | b7fd084ab5b2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | tied | better | graph_prior_only | 40 | 3fd032a07a14 | d958d4651249 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | tied | better | graph_prior_only | 40 | c809cf0873dc | 90ac3e87e9a6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | better | better | learned_route | 100 | 9e05bcdb498f | fd2200c94602 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | tied | better | graph_prior_only | 100 | 4a7d8d22f18a | f12611891962 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | tied | better | graph_prior_only | 40 | 3ed009db5884 | 866e119a3062 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | tied | better | graph_prior_only | 40 | 6af29aac8fe5 | 11baaa1bb3a6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | tied | better | graph_prior_only | 40 | bde5464a660d | 547b9ce8ef6e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | tied | better | graph_prior_only | 40 | fccdb3ac2b5b | 71cf316f0b29 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | tied | better | graph_prior_only | 40 | af42220501d7 | 6caeb0864c4a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | tied | better | graph_prior_only | 40 | 09036b84eba3 | 9ee15c3da2fe |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | tied | better | graph_prior_only | 40 | d8175b0b90ee | e67048705ebb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | tied | better | graph_prior_only | 40 | 7f069c9b49d8 | b2e48fba9ae2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | tied | better | graph_prior_only | 40 | 87f84e6fb60c | dcc8c6333ef7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | tied | better | graph_prior_only | 40 | ec072d091b8f | 95b400e80ced |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | tied | better | graph_prior_only | 40 | 17b64aadfe77 | 8aef0fbe0ff5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | tied | better | graph_prior_only | 40 | a7a22775e2e7 | f014bfdfc5b7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | tied | better | graph_prior_only | 40 | ee5486915998 | 5d2048260f8b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | tied | better | graph_prior_only | 40 | 8d752285ad6a | 4e1cd52286f7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | tied | better | graph_prior_only | 40 | 3f9cd64e7bd2 | 466a86f60fa0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | tied | better | graph_prior_only | 100 | a56564b45439 | dac30a717db6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | tied | better | graph_prior_only | 40 | e5e0d0fb86bf | 5a473921a7d9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | tied | better | graph_prior_only | 40 | 718d7bce6ab3 | 11d8195a4e57 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | tied | better | graph_prior_only | 40 | 78c5937bfda5 | a79a99eba81d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | tied | better | graph_prior_only | 40 | 37925d001a10 | d21b7573467b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | tied | better | graph_prior_only | 40 | 96a9b92a26a8 | a05b66d25324 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | tied | better | graph_prior_only | 40 | d37c8d47ec1e | f310994f3609 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | tied | better | graph_prior_only | 40 | 360854437613 | d1e22fae7c19 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | tied | better | graph_prior_only | 40 | a2af81b49679 | 3c12256687f5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | tied | better | graph_prior_only | 40 | 0353ada2db6e | d2598e2a753d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | tied | better | graph_prior_only | 40 | 9fea584af19b | 85332109cd64 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | tied | better | graph_prior_only | 40 | 98aecc519749 | e3122797dcf0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | better | better | learned_route | 100 | 6c1517fdf549 | 00ca9dbb0a99 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | tied | better | graph_prior_only | 40 | 473bb77f4173 | 728dac5d783e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | tied | better | graph_prior_only | 40 | 5446d3fdbdcb | 452f982e4c1e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | tied | better | graph_prior_only | 40 | 33f76166cfae | a9d440336fa9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | tied | better | graph_prior_only | 40 | 20fe8018c08b | 910d6349a873 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | tied | better | graph_prior_only | 40 | 6d667767ed40 | 988c30734b27 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | tied | better | graph_prior_only | 40 | a368ee0e7535 | fd4fc6dbf95b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | tied | better | graph_prior_only | 40 | 14e7dbdeffcd | e9b641564411 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | tied | better | graph_prior_only | 40 | 0d062afa3a6b | 03f0683cdfd1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | tied | better | graph_prior_only | 40 | 93893b53e327 | 55e1d3668f63 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | tied | better | graph_prior_only | 40 | 6a93478dbd58 | 16bb8454fef3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | tied | better | graph_prior_only | 40 | 278cbd9f7782 | f3133e243c63 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | tied | better | graph_prior_only | 40 | 33221157c2cd | d97da3c38b87 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | tied | better | graph_prior_only | 40 | 00f60ea0f431 | 7c89560d337a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | tied | better | graph_prior_only | 40 | bee77e2e7921 | 7124283fa969 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | tied | better | graph_prior_only | 40 | 2d58212bf19f | 2a5016f8ca24 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | tied | better | graph_prior_only | 40 | 645f81ec6435 | a6d1c57bec0b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | tied | better | graph_prior_only | 40 | 35f5ee503e92 | 86bf2f4a9dff |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | tied | better | graph_prior_only | 40 | f72b378dc25b | 8731ecfbef42 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | tied | better | graph_prior_only | 40 | 455e16c2233a | 74a8b83405a0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | tied | better | graph_prior_only | 40 | 2c0b7ce14221 | 538afc4579a3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | tied | better | graph_prior_only | 40 | 9bd06fb704b8 | 4d8ce3309089 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | tied | better | graph_prior_only | 40 | 2a489c9f2c4e | 383503fcf343 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | tied | better | graph_prior_only | 40 | a87e118d1804 | 07eacfe0c10f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | tied | better | graph_prior_only | 40 | e0bf65c2668b | bdc36ced1c0b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | tied | better | graph_prior_only | 40 | 1fc658bab003 | a3ae2eeb15ee |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | tied | better | graph_prior_only | 40 | ba95bd409d00 | ec01ee0d6514 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | tied | better | graph_prior_only | 40 | a6ba73939222 | 3a7acdc23ac7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | tied | better | graph_prior_only | 40 | 1a0d319139c8 | b2ca26ce5e05 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | tied | better | graph_prior_only | 40 | 944842c45b3f | 7cf293d4e2e7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | tied | better | graph_prior_only | 40 | f11a775a91f0 | 49bda3aba155 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | tied | better | graph_prior_only | 40 | c37adcfaf685 | f65947b9e716 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | tied | better | graph_prior_only | 40 | e69b73a24e67 | e0c8c7687519 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | tied | better | graph_prior_only | 40 | eafbcfebcc0c | 227273b2bb06 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | tied | better | graph_prior_only | 40 | 9596c14941bd | 1247dd084b84 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | tied | better | graph_prior_only | 40 | 6bc48c0b2bd3 | 9d320bed4cbb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | tied | better | graph_prior_only | 40 | 5d19e3972e29 | 225580974bab |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | tied | better | graph_prior_only | 40 | 02eb5f429255 | a0df3ef22557 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | tied | better | graph_prior_only | 40 | 43628cadf9cd | c9687fd8d462 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | tied | better | graph_prior_only | 40 | 3459323df552 | 2b18084b99e3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | tied | better | graph_prior_only | 40 | d39317d4397d | eb2701606b2f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | tied | better | graph_prior_only | 40 | e23c22526872 | 0ae8e8c26f07 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | tied | better | graph_prior_only | 40 | f9c0571e0da0 | 51c8a1b23612 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | tied | better | graph_prior_only | 40 | d040ad32cfa7 | 6ed43ab4ce1a |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | tied | better | graph_prior_only | 40 | 6a8012db2cae | 4da2fd43374f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | tied | better | graph_prior_only | 60 | 1a0181588fdb | 2a1e0d770323 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | tied | better | graph_prior_only | 60 | ae32f3bb74bb | 7741d4fce06f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | tied | better | graph_prior_only | 40 | 3ab9bde1da52 | f2cbbacfe117 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | tied | better | graph_prior_only | 40 | e5a58a12c1d3 | 6e425a8fc8d7 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | tied | better | graph_prior_only | 40 | dc066bc01170 | 7487c29dd698 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | tied | better | graph_prior_only | 40 | 2437b117568d | 6d0cb7ded96e |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | tied | better | graph_prior_only | 40 | d7fccb323e0c | d04acab2df9f |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | tied | better | graph_prior_only | 40 | c1bea6a79d68 | aae5e92585e6 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | tied | better | graph_prior_only | 40 | 5266ca210500 | 953812267fe0 |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | tied | better | graph_prior_only | 40 | 12ab8ac897ec | 5fdbff63df48 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | tied | better | graph_prior_only | 60 | 04ef82cfde93 | 975318ca12b2 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | tied | better | graph_prior_only | 40 | 543da087e7d1 | 182d640921be |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | tied | better | graph_prior_only | 40 | 8d1bc09479a8 | 1b1b18c53e5c |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | tied | better | graph_prior_only | 60 | 778faf114fb3 | 7e949ffe415d |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | tied | better | graph_prior_only | 40 | cb8ef234df26 | ec4e4ad336b6 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | tied | better | graph_prior_only | 40 | cb7e13b59660 | 90febdb47b3a |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | tied | better | graph_prior_only | 40 | dfac9c89611b | 7f23b625c865 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | tied | better | graph_prior_only | 40 | 049f0f5e1f48 | fb10a6dcc603 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | tied | better | graph_prior_only | 40 | 4e9af4ebbfec | 58eaffec460d |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | tied | better | graph_prior_only | 40 | 535bdcc1094d | 828c9319d3d8 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | tied | better | graph_prior_only | 40 | 290d99ef2d14 | 79269cec636b |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | tied | better | graph_prior_only | 40 | 724ea919dea4 | d632cb7e9357 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | tied | better | graph_prior_only | 40 | 5e256771c9a4 | 0f07ce9be4a7 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | tied | better | graph_prior_only | 40 | 3cdfe1a3e6a0 | af49291fe81f |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | tied | better | graph_prior_only | 40 | 134df7de53fe | 25098fb7b345 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | tied | better | graph_prior_only | 40 | 1cb77b89bb1f | 74ce6aa06a13 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | tied | better | graph_prior_only | 40 | 93612d257496 | fc995dac6728 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | tied | better | graph_prior_only | 40 | 079fa703eedf | e8669d5361e8 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | tied | better | graph_prior_only | 40 | 5204eed7419e | 44df0b71d13c |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | tied | better | graph_prior_only | 40 | f6bd5763b8e6 | 2f5c69f9a8f6 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | tied | better | graph_prior_only | 40 | 8c490267d950 | cd7f5f749792 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | tied | better | graph_prior_only | 40 | f73021355774 | 1b3f44a925ec |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | tied | better | graph_prior_only | 40 | cb2468e39426 | be19017e5deb |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | tied | better | graph_prior_only | 40 | 63d3afc7629e | 9ca6c4b3e286 |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | better | better | learned_route | 60 | e920c9a3732d | 3587f5ae685e |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | tied | better | graph_prior_only | 40 | 90c8689acf79 | 36cfc613db99 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | tied | better | graph_prior_only | 40 | 553978f99929 | d3ba5178bb28 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | better | better | learned_route | 70 | 7176d9aeef83 | 968ce72326dc |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | tied | better | graph_prior_only | 40 | 67b76981513f | 39c32b12cd0a |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | tied | better | graph_prior_only | 60 | 8cb14ffc761e | 9ecfd0aa06aa |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | tied | better | graph_prior_only | 40 | d8b129e01784 | be4a3149eabf |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | tied | better | graph_prior_only | 40 | a076cd8a6827 | 118226cd4826 |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | tied | better | graph_prior_only | 40 | 889d1c09b029 | 0b6bfb0fcb45 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | tied | better | graph_prior_only | 40 | f5446545b333 | 1afa2dbd9006 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | tied | better | graph_prior_only | 40 | 5b5dbe5421a8 | a366b10918e2 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | tied | better | graph_prior_only | 40 | 73aff5f06ce3 | 98231227dc7f |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | tied | better | graph_prior_only | 40 | da9409b7e2a8 | 7e81889f0329 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | tied | better | graph_prior_only | 40 | 61b4e4cfd50f | dfd58ad43a59 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | tied | better | graph_prior_only | 40 | fb3c37961bd8 | cd02e661da95 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | tied | better | graph_prior_only | 40 | 836666779c8b | a79329b6aff0 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | tied | better | graph_prior_only | 60 | 23a2fbc9a99c | 3d7b440ad07d |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | tied | better | graph_prior_only | 60 | 1bf5238e9b64 | 36088425b6e7 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | tied | better | graph_prior_only | 40 | 43a5b87746dd | 7125be588bf4 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | tied | better | graph_prior_only | 40 | 653995fa5e0d | 65a4eb91c0e3 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | tied | better | graph_prior_only | 40 | 5981dce1c4e6 | b32f5f00dceb |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | tied | better | graph_prior_only | 40 | 2ff65bb385ea | a6bf306c462d |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | tied | better | graph_prior_only | 40 | 098ec58f5b83 | 24be08d0d9b7 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | 392375373d06 | d51bedd0fffe |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | tied | better | graph_prior_only | 40 | ead0aeaf3a05 | c9ee36de1908 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | tied | better | graph_prior_only | 40 | 4824910b157a | b7e1124fc5a1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | tied | better | graph_prior_only | 40 | 5ebad42a4b03 | 68f5b33c7c31 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | tied | better | graph_prior_only | 40 | b84b40847e2d | 807765a1d8e4 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | tied | better | graph_prior_only | 40 | 3fbdb6e45f90 | 98cdac3f17a2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | tied | better | graph_prior_only | 40 | 4591b8d0ccae | d6ca0f0c4112 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | tied | better | graph_prior_only | 40 | 055a5c99ede5 | 4e74bcd1562a |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | tied | better | graph_prior_only | 40 | 913c72348465 | 268168280370 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | tied | better | graph_prior_only | 40 | 2c512ab221e0 | b6c94aab3b55 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | tied | better | graph_prior_only | 40 | dc000234f7bb | 8efbefdd9c54 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | tied | better | graph_prior_only | 40 | 2c3f3006a698 | 2122d5c91f3f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | tied | better | graph_prior_only | 40 | 6a68e0bbe0a1 | 265223b2af68 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | tied | better | graph_prior_only | 40 | e67a0d77c41d | 7f82d0a5bd8d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | tied | better | graph_prior_only | 40 | eac4c3bedf10 | 6957e57a4ddc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | tied | better | graph_prior_only | 40 | d6e0fe70b2d3 | cc8c40cfbbf2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | tied | better | graph_prior_only | 40 | 854174e7f2bd | 07b42387a2c9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | tied | better | graph_prior_only | 40 | e2246c12c078 | cc97874c1c31 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | tied | better | graph_prior_only | 40 | 5cb3f0071308 | 30f06ad3b33f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | tied | better | graph_prior_only | 40 | 034b42514b45 | 453df7140ad5 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | tied | better | graph_prior_only | 60 | 6679f700db3c | 81e86168d276 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | tied | better | graph_prior_only | 40 | 34c42f730543 | 59b7d46218cf |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | tied | better | graph_prior_only | 40 | aabe407e02b1 | 5094e5a9e7dd |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | tied | better | graph_prior_only | 40 | c8141b372e98 | a1d8e9a12d24 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | tied | better | graph_prior_only | 40 | c7b342a3a795 | 9f33c58dcb6d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | tied | better | graph_prior_only | 40 | 089fc19c9c7e | 0c69a5bc5aed |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | tied | better | graph_prior_only | 40 | e72db12a2c73 | ee2a90e7049b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | tied | better | graph_prior_only | 40 | c2a8498d7ba4 | 5dc4ddd8fd5e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | tied | better | graph_prior_only | 40 | f51fce3be2bf | 9ac810969f6c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | tied | better | graph_prior_only | 40 | c2f703e93b52 | 7259f7520355 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | tied | better | graph_prior_only | 40 | 2c05abe31da2 | f771d54fd3b7 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | tied | better | graph_prior_only | 40 | 0a472802684f | 661c28de3960 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | tied | better | graph_prior_only | 40 | cedf3d5e5487 | 27bd25a5f480 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | tied | better | graph_prior_only | 80 | 51efb8bdd9be | a64957333848 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | tied | better | graph_prior_only | 80 | 70f179191f36 | dafcb59f2fe8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | tied | better | graph_prior_only | 40 | 50f29a0361b1 | 729e0d55f579 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | tied | better | graph_prior_only | 80 | 269b45ff0767 | 09c7c2716c6c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | tied | better | graph_prior_only | 40 | 6e3e3b709eca | 0aaad155171e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | tied | better | graph_prior_only | 40 | cb836ddc23e0 | bd051165d107 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | tied | better | graph_prior_only | 40 | cda8ff2b92c3 | 9a3f9e324aa4 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | tied | better | graph_prior_only | 40 | 07a5d218eae7 | c16084d6d38a |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | tied | better | graph_prior_only | 40 | 2cdfe668626a | b52617649538 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | tied | better | graph_prior_only | 40 | 66742677fa4b | 5d33b97456cf |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | tied | better | graph_prior_only | 40 | 2e3faeba4e9e | 4321a597bc38 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | tied | better | graph_prior_only | 40 | dbf7d618e57d | 9bb7e07cac1d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | tied | better | graph_prior_only | 40 | ca6a7fd74cfd | 03cdcd8fa06e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | tied | better | graph_prior_only | 40 | d31e96fd9713 | 159739120c50 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | tied | better | graph_prior_only | 40 | 82febe40d1d3 | fe90d6ac3f89 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | tied | better | graph_prior_only | 60 | 688bfb15c38c | d2f922b822db |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | tied | better | graph_prior_only | 40 | 47bcffdee19f | 5bbf1811e04b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | tied | better | graph_prior_only | 60 | 2908015b9926 | 01e86a6a0969 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | tied | better | graph_prior_only | 40 | 085cba30a1f2 | a6d2c6777074 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | tied | better | graph_prior_only | 40 | 802e59bfd2f6 | 8a59e044e00f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | tied | better | graph_prior_only | 70 | 38f7c98d3c95 | 9ece817185fd |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | tied | better | graph_prior_only | 40 | 36acf0e114fd | 2991be5ac98a |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | tied | better | graph_prior_only | 40 | 6c28d2bb9b51 | 8584bdde94b3 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | tied | better | graph_prior_only | 40 | 8e6631a512f5 | 8d70a7c1b18e |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | tied | better | graph_prior_only | 40 | e3a5fe0dbc05 | b1ffe4fce09d |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | tied | better | graph_prior_only | 60 | fedb43a5f2f3 | e2fc20e37ad8 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | tied | better | graph_prior_only | 40 | ae310853c472 | d72bd6a37974 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | tied | better | graph_prior_only | 40 | d849260ac1c5 | aa4ce0f182ec |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | tied | better | vector_only | 60 | 32553245d865 | 8909bc812d06 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | tied | better | graph_prior_only | 40 | d299d2515984 | 998f9d969e19 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | tied | better | graph_prior_only | 70 | 231d6bf270ac | f28c2bad4c72 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | tied | better | graph_prior_only | 40 | d3d22f8fb946 | 03158b39663e |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | tied | better | graph_prior_only | 40 | aa660adb7fc7 | f6068e8c10c2 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | tied | better | graph_prior_only | 40 | aef2689a509b | cd4ea436ff3f |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | tied | better | graph_prior_only | 40 | eaac26255094 | b531fd3efe00 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | tied | better | graph_prior_only | 40 | f133b7e9e7f1 | 7ff41023a390 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | tied | better | graph_prior_only | 40 | 93babb72b718 | 6d05ca8eb7cd |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | tied | better | graph_prior_only | 40 | ef8ff4582435 | 1ea7cc98fb51 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | tied | better | graph_prior_only | 40 | 4fa723f58b8f | d1c69068e69e |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | tied | better | graph_prior_only | 40 | 0a05e3d44fe7 | f851fdd8c096 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | tied | better | graph_prior_only | 40 | 3e5d6aa47ae0 | 541303554e2b |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | tied | better | graph_prior_only | 40 | f946d99c05a1 | 61881cca765b |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | tied | better | graph_prior_only | 70 | e9e157ee8e8e | 46b64185a987 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | tied | better | graph_prior_only | 40 | 6840db920f2e | a440ebab3c30 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | tied | better | graph_prior_only | 40 | f69cf08c082c | 8aa57d595830 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | tied | better | graph_prior_only | 40 | 874fb79035c8 | 53b8a60e3f89 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | tied | better | graph_prior_only | 40 | 32dcd85105c0 | 04771df094b1 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | tied | better | graph_prior_only | 40 | d33573ef3dc3 | abd209c2d861 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | tied | better | graph_prior_only | 40 | 035b2ff6132d | efbcf2b07b9a |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | tied | better | graph_prior_only | 40 | 943e9c9d8779 | aede5bd74397 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | tied | better | graph_prior_only | 40 | 587210b7bd66 | 2f27090b8626 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | tied | better | graph_prior_only | 40 | fefa94bf4ad2 | 3a96f1a10bda |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | tied | better | graph_prior_only | 40 | 0209ac28e594 | 81e48f90c80c |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | tied | better | graph_prior_only | 40 | 8a405150ba26 | f800bd46a2da |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | tied | better | graph_prior_only | 40 | e502725fa7e0 | 0de087ac263a |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | tied | better | graph_prior_only | 40 | 4e9c47743ce8 | 2f0be4259955 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | tied | better | graph_prior_only | 40 | 476f9523961b | a5a9675e70f5 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | tied | better | graph_prior_only | 40 | 9aff3ade2b3d | e0ef99227eca |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | tied | better | graph_prior_only | 40 | ff7939858c66 | 700f49641e00 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | tied | better | graph_prior_only | 40 | d625ad7e22af | 809cae149692 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | tied | better | graph_prior_only | 40 | dde3f49bf874 | caed469a643a |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | tied | better | graph_prior_only | 40 | f2aa7d2fe0d7 | a6501d329f32 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | tied | better | graph_prior_only | 40 | 7f8d0dfa9f2b | d755e284b013 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | tied | better | graph_prior_only | 40 | fa803ecc9bbc | ba1f12c62182 |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | tied | better | graph_prior_only | 40 | 1865eddb92dc | 17f432c2dabb |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | tied | better | graph_prior_only | 40 | c4cb9030668f | 37a00faf7990 |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-35fd34d9ef839d94bb68e0b8af93399d1d797d411dfea8121b0d3e51f459efd9 |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-cc79e13bd83b1e7b40212079f4c23e7d735fc1164edecc84a22777544dd0bab3 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-a5b79da1fb8eb9fe03502e5bb0d2515dbfd6146c2738765f8859f3293fc83dfe |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-fc1d5e4201a12091552179f5df5de7bf107c49e20d47787c6a60983c92329d65 |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ef92288600cf173dffb4e053fa6fbe21ee73855f99dc277060f75acbba0e9a71 |
| worked-traces | worked-traces.md | none | sha256-2530e0bfd42850eeda32f7c2e57cc8c89e8505ba72f4a7d05516d9cedece48c2 |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-fc9f05affe6c3f922a6f3a306819892bd46590c376a8cddb1d2e4ecaf4b605a1 |
