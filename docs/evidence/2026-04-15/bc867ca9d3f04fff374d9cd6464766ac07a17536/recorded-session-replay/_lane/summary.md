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
- learned_route vs graph_prior_only (traces): 9 better, 394 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 403/403 (1)
- learned_route vs graph_prior_only (turns): 9 better, 394 tied, 0 worse
- regressions vs graph_prior_only: 0/403 (0)
- regressions vs no_brain floor: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 68/832 required-context phrases vs graph_prior_only 57/832
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- success-adjusted economics: success-adjusted economics are not computed in replay-lane aggregates; use comparative eval or proof-cron for prompt-cost proxy surfaces
- fail-open: fail-open posture is not modeled in recorded-session replay lane aggregates; use proof-cron health surfaces for degraded-serve reporting

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 0 | 403 |
| graph_prior_only | 394 | 394 |
| learned_route | 9 | 403 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | tied | better | graph_prior_only | 60 | cf15e98fab40 | 8ce020736a1a |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | tied | better | graph_prior_only | 40 | 14977f5ac211 | 8c4ba19c0324 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | tied | better | graph_prior_only | 70 | 0df4addc5970 | 4dfef9f260be |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | tied | better | graph_prior_only | 80 | 267fed6ac36a | 33852c7f42b9 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | tied | better | graph_prior_only | 40 | b926fbe247b1 | 00ab279004c6 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | tied | better | graph_prior_only | 60 | 9de6b2b247eb | 6554f455edb4 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | tied | better | graph_prior_only | 60 | 66f4264a24b5 | 1fb623f298df |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | tied | better | graph_prior_only | 60 | 0f13d36d27c1 | 4db1779b291b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | tied | better | graph_prior_only | 60 | ef9c2862b477 | b84935cca1de |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | tied | better | graph_prior_only | 40 | 30fb7bc7b654 | 7c8c314fc553 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | tied | better | graph_prior_only | 60 | 600ed322cfd8 | 60de269c423b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | tied | better | graph_prior_only | 40 | acc62b67c27f | fd0d328da795 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | tied | better | graph_prior_only | 40 | 85b1d9cc1353 | 28a34dc31c40 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | tied | better | graph_prior_only | 40 | 049d22a1f8b9 | 536dfcd009bb |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | tied | better | graph_prior_only | 40 | b294d512e224 | 5b1a21340017 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | tied | better | graph_prior_only | 40 | 5ec438f56dfd | c39c5dd96c37 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | tied | better | graph_prior_only | 60 | 4086aa51de44 | fca5d49daff2 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | tied | better | graph_prior_only | 40 | cf3fc4c4aff6 | 5b09a90a9e1b |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | tied | better | graph_prior_only | 40 | de349710368f | f90cbdf43130 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | tied | better | graph_prior_only | 40 | 3738a5dbe2e6 | 283ed521162f |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | tied | better | graph_prior_only | 40 | b38642163da0 | 08f5693e0546 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | tied | better | graph_prior_only | 100 | 70a1164dfb97 | 0af5cf6bc5bf |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | tied | better | graph_prior_only | 40 | 0a7a9a063405 | 00bd60a6a6c8 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | tied | better | graph_prior_only | 40 | 8760a72a7919 | f8693e8ba188 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | tied | better | graph_prior_only | 100 | c684145354e8 | de783da05421 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | tied | better | graph_prior_only | 40 | f670513d9cdb | e02869bd0c9f |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | tied | better | graph_prior_only | 40 | e35b7a89d4a6 | a8074f31d68b |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | tied | better | graph_prior_only | 100 | c1cb0f527513 | 6a1408c23999 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | tied | better | graph_prior_only | 40 | 9e674fd6af58 | 7b2bd1ff22e4 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | tied | better | graph_prior_only | 40 | 1d8ebd2e75cb | f0d76651d805 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | tied | better | graph_prior_only | 40 | 04ef2fc39019 | 1293185f63f6 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | tied | better | graph_prior_only | 40 | 6fcbb3f1a413 | 8f916521fdc3 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | tied | better | graph_prior_only | 60 | faceb0b58d94 | 7a1964f710b6 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 789544c5464d | 8a9fabe00d17 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | tied | better | graph_prior_only | 40 | 3dcc09baec7a | 15a141e5d32c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | tied | better | graph_prior_only | 40 | ef499aadfc89 | ec0c1e6e893f |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | tied | better | graph_prior_only | 40 | 8d85c92cca6d | 9a3a97b8850e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | tied | better | graph_prior_only | 40 | d7f7c320c7f2 | abc663340948 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | tied | better | graph_prior_only | 40 | 5f2baa3d0cef | 6162b176c54a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | tied | better | graph_prior_only | 40 | 3abe57b4fba2 | 2a5ab7a09434 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | tied | better | graph_prior_only | 40 | 0b6bd1fb0c75 | ba94cbdc1eea |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | tied | better | graph_prior_only | 40 | 3580c22134e3 | 6308c7853401 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | tied | better | graph_prior_only | 40 | d81321f7906d | 9ca83ed1b603 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | tied | better | graph_prior_only | 40 | ecb1df4251d1 | 913b6155e38a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | tied | better | graph_prior_only | 40 | df5527426f9d | 41e8dc2d01bf |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | tied | better | graph_prior_only | 40 | 98d89cad31ce | 31c36cd11073 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | tied | better | graph_prior_only | 40 | aa3232c18ebb | f7a3e1da4119 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | tied | better | graph_prior_only | 40 | 22af5110ae89 | 369749d617e3 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | tied | better | graph_prior_only | 40 | 3d14fee0d55c | 535ae677f86c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | tied | better | graph_prior_only | 40 | 5aa1c688e4b1 | e2edd4902374 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | tied | better | graph_prior_only | 60 | 2674e7fbcc0f | 29d38d08e777 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | tied | better | graph_prior_only | 40 | 0b1dbca9fd3b | a994a11bfb0e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | tied | better | graph_prior_only | 40 | 3b07e563d698 | 68b977cf7b6e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | tied | better | graph_prior_only | 60 | a57523bd0c41 | 9e1121794e8c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | tied | better | graph_prior_only | 40 | 7e032fd7672e | 2a04dce6a906 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | tied | better | graph_prior_only | 40 | ec89c1ac372c | 8c5b3d0e90e3 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | tied | better | graph_prior_only | 40 | f0a5e9e3c639 | ee8e6eb966b8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | tied | better | graph_prior_only | 40 | ed2353618059 | 00d45e094ad3 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | tied | better | graph_prior_only | 40 | 7746989f94a4 | 85b79a767548 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | tied | better | graph_prior_only | 40 | 86c945f1a816 | a11163a6f80a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | tied | better | graph_prior_only | 40 | fced42a8bea1 | bab5aa0dce7d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | tied | better | graph_prior_only | 40 | 50fd3bcfe616 | e637b025cd58 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | tied | better | graph_prior_only | 40 | 849f78182f78 | dcd00aa27534 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | tied | better | graph_prior_only | 40 | 3f815fd01bc4 | d1286583c045 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | tied | better | graph_prior_only | 40 | 0fd7c5fb41e5 | 37320736182c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | tied | better | graph_prior_only | 40 | 730bc8e8ab06 | 2036b1a1356e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | tied | better | graph_prior_only | 40 | e5499f8d8ee6 | 97ef10669676 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | tied | better | graph_prior_only | 40 | af125dccd799 | da096f5f8285 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | tied | better | graph_prior_only | 60 | a1e73c519572 | f4a3a9abe49e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | tied | better | graph_prior_only | 40 | 0fd7e4488c8a | c30942cd5148 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | tied | better | graph_prior_only | 40 | 477e8393dd43 | 1ef4a8edc778 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | tied | better | graph_prior_only | 40 | 756327952b55 | 8617588ca54f |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | tied | better | graph_prior_only | 40 | 5e79eee76093 | 3984155d52f4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | tied | better | graph_prior_only | 40 | f1a4e00346bc | 60402e1b703b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | tied | better | graph_prior_only | 40 | 83f7c3478b5b | 6d0771e2e1ff |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | tied | better | graph_prior_only | 40 | 02ae80bb8e08 | fd09b3de1a9d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | tied | better | graph_prior_only | 40 | ed1532293414 | 2fe4943e18e9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | tied | better | graph_prior_only | 60 | dab42176a148 | 82a26127d459 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | tied | better | graph_prior_only | 40 | 174c4774ad12 | 2224d081ae2a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | tied | better | graph_prior_only | 40 | 74bc830b6eb6 | 89ba1383e500 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | tied | better | graph_prior_only | 40 | 4e9907a04f4a | 2340d2a431a4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | tied | better | graph_prior_only | 40 | 5c66c43c70d4 | 75bb2f6a8329 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | tied | better | graph_prior_only | 40 | d4c8448dffe4 | 7d697a491520 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | tied | better | graph_prior_only | 40 | 2586d64d8717 | cac9f1028b7e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | tied | better | graph_prior_only | 40 | cfccf926ca0e | db5ace593510 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | tied | better | graph_prior_only | 40 | 5f29a7af3474 | 2a6d68f45819 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | tied | better | graph_prior_only | 40 | 1d71943d01c6 | 1a2d108189aa |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | tied | better | graph_prior_only | 40 | 07c1ac0e44f4 | 0b9fc2ecee50 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | tied | better | graph_prior_only | 40 | 82a753e36965 | 17dbf9a407cd |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | tied | better | graph_prior_only | 40 | b704caf8ca39 | 17c94a5fa980 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | tied | better | graph_prior_only | 40 | fe56d7eb846a | c797fb5de8b9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | tied | better | graph_prior_only | 40 | 42be2d867930 | 4adc44f4c8bb |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | tied | better | graph_prior_only | 40 | 930c1550e762 | 525af610be61 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | tied | better | graph_prior_only | 40 | abba3615ab41 | 44e4c3783630 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | tied | better | graph_prior_only | 70 | 7477b2dd5045 | 8614300975ac |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | tied | better | graph_prior_only | 40 | 6cd764fa8641 | 74c8423c9ea8 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | tied | better | graph_prior_only | 40 | ba2fbadf7637 | cb600391abf8 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | tied | better | graph_prior_only | 40 | a2952dcbb569 | 610577fa10a3 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | tied | better | graph_prior_only | 40 | 94f89f9602f6 | 0e0da30e36c4 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | tied | better | graph_prior_only | 40 | 91fe5c810efe | 4897b0115d1a |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | tied | better | graph_prior_only | 40 | 5577204bef58 | 14cda44b8f17 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | tied | better | graph_prior_only | 40 | 9abc689ae65a | 1805b8ea27de |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | tied | better | graph_prior_only | 40 | a08d13a7629c | 2235d554c97d |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | tied | better | graph_prior_only | 40 | 9f53a4872aa4 | aa011b0f7f3d |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | tied | better | graph_prior_only | 40 | 242946857faa | 489799966550 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | tied | better | graph_prior_only | 40 | bddbbb6f13f1 | b3dcf0a2f8d5 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | tied | better | graph_prior_only | 40 | b028b80499ed | 0d507f250852 |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | tied | better | graph_prior_only | 40 | 36c24f32e8f9 | 3c2d0f4893b0 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | tied | better | graph_prior_only | 40 | a68c40374ba8 | d1e0ef26efbf |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | tied | better | graph_prior_only | 40 | 5ebb590267fb | 84131156b111 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | tied | better | graph_prior_only | 60 | 70310113c02e | ed66518531f8 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | tied | better | graph_prior_only | 40 | 41620662d99b | 93005ed19c41 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | tied | better | graph_prior_only | 40 | 41c67933cfba | 21a40185e9fd |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | tied | better | graph_prior_only | 40 | 70a51d5944f0 | e93ae966b85b |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | tied | better | graph_prior_only | 100 | a449effd97ba | b639ba57fe44 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | tied | better | graph_prior_only | 40 | f4a9868fcce8 | 50604cabd83e |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | tied | better | graph_prior_only | 40 | 41797db6e337 | 6e6184469f19 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | tied | better | graph_prior_only | 40 | 57e837497705 | 1a56d7ef754d |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | tied | better | graph_prior_only | 70 | 504956747f33 | b18f9fb4313b |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | tied | better | graph_prior_only | 40 | 3f26d1b2e775 | 4f5ee28dfdea |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | tied | better | graph_prior_only | 40 | 5c1d0f592e13 | 7c1c0667b46a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | tied | better | graph_prior_only | 40 | 01428b823764 | e525e66a0a9f |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | tied | better | graph_prior_only | 40 | b968afd22a51 | deca7d6216a8 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | tied | better | graph_prior_only | 40 | 3b1c5202e7bb | ce6cff090ced |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | tied | better | graph_prior_only | 40 | 6d889cc9e7eb | c70dab9fd8f4 |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | tied | better | graph_prior_only | 40 | c05b333dc511 | 013928a836e5 |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | tied | better | graph_prior_only | 100 | b1f01f123b5c | 0300697c8588 |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | tied | better | graph_prior_only | 40 | 55b5bfefe124 | a17e79605c50 |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | tied | better | graph_prior_only | 40 | d481d5ba91dc | 7bc18e984ad2 |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | tied | better | graph_prior_only | 40 | 2303ff3d2fcb | 8ef66c6afe93 |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | tied | better | graph_prior_only | 80 | 5dcf937c37cb | 87dc69854888 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | 2e3cf8bddb0b | 3d35b73892d4 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | tied | better | graph_prior_only | 40 | ace5d8cec38e | 1835b8b4c135 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | tied | better | graph_prior_only | 60 | 8938119e7eb1 | 24c8e5d95cd5 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | tied | better | graph_prior_only | 40 | e2eaac0b0f21 | 757baeecf408 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | tied | better | graph_prior_only | 40 | f9081fe72567 | 1b37bfbcc0f0 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | tied | better | graph_prior_only | 40 | bb2575e1152f | bbe70bdd54d4 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | tied | better | graph_prior_only | 40 | b6da0e3c364e | 2a1cb78c66e1 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | tied | better | graph_prior_only | 40 | f4bd5ee9f9a5 | 0954809d2b87 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | tied | better | graph_prior_only | 40 | d32551f5a503 | 1025f6b5a221 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | tied | better | graph_prior_only | 40 | 7580a87f8a86 | 169af59d07c3 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | tied | better | graph_prior_only | 40 | 63306b97ef0c | 3995ba16c8f0 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | tied | better | graph_prior_only | 40 | 575d5f789fa5 | bfe50bb2d3d4 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | tied | better | graph_prior_only | 40 | 80a1e4c747ba | 404d9ccb3a52 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | tied | better | graph_prior_only | 40 | 8c8c69507588 | 95860c927644 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | tied | better | graph_prior_only | 40 | a94311bf991a | 60986be37ab7 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | tied | better | graph_prior_only | 40 | 2a3a19e105a2 | dc758f26b852 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | tied | better | graph_prior_only | 40 | 94f1a7a837fc | b4227ec674c2 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | tied | better | graph_prior_only | 40 | 91e96c1997fd | 3310d72e3095 |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 28ea7f8c5a64 | 5a10f6a02df4 |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | better | better | learned_route | 80 | 8deb42ddf98f | d6ae1ea78c5b |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | tied | better | graph_prior_only | 60 | ea00070f520c | 23dab062268a |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | tied | better | graph_prior_only | 80 | dafd07099551 | 84828fa6b368 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | better | better | learned_route | 80 | ee165ccc126d | 0b4cd9fda769 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | better | better | learned_route | 60 | 5fb9fef27d23 | a85684f5af46 |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | tied | better | graph_prior_only | 40 | 5a13798cd802 | 8a26414a91c7 |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | tied | better | graph_prior_only | 40 | 874e889da065 | 4f5a8ff64fba |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | tied | better | graph_prior_only | 40 | 465da4892a11 | 5896c729b19d |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | tied | better | graph_prior_only | 60 | 45726b27eea7 | 330970322417 |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | tied | better | graph_prior_only | 40 | ed358ed9c1e9 | 21ddf7e91a65 |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | tied | better | graph_prior_only | 100 | b7a51ec984d6 | 773694158527 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | better | better | learned_route | 70 | 29add3b3ae82 | 9ae98292a02a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | tied | better | graph_prior_only | 40 | b24a8eedcab6 | f4ad7b9293c9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | tied | better | graph_prior_only | 40 | 60e609990f3b | c951b08e1814 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | tied | better | graph_prior_only | 40 | e97e8f861e73 | 96cd35e9ca79 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | tied | better | graph_prior_only | 40 | 1d13271642cd | 0ab4f8ea9e27 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | tied | better | graph_prior_only | 40 | 8ca4a7bc4e84 | fb98adafe1de |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | tied | better | graph_prior_only | 40 | 42146f8b80b0 | f073931d8aa1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | tied | better | graph_prior_only | 40 | f53e2f4ace2b | 6e36ad633455 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | tied | better | graph_prior_only | 40 | 9eeece0e71af | 024ecf29cacb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | tied | better | graph_prior_only | 40 | 16e19d3f6514 | 24a2e7889f0d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | tied | better | graph_prior_only | 40 | 1514cafe8b2f | 27219c0c5bfb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | tied | better | graph_prior_only | 40 | 454b4da02701 | b3f255d939e7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | better | better | learned_route | 100 | cb7adf23e036 | 3f2b6c3ab476 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | tied | better | graph_prior_only | 100 | 706e36d3fbd8 | 62a5cbb17dc7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | tied | better | graph_prior_only | 40 | fd8eb6715dcc | 65351576be6d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | tied | better | graph_prior_only | 40 | a5b98358317c | a06810bbdab4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | tied | better | graph_prior_only | 40 | 75b11962c240 | f4765749e9c5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | tied | better | graph_prior_only | 40 | 3d23400b4edc | 289241e05ace |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | tied | better | graph_prior_only | 40 | 9c589aee9bdb | 3c9dbb482dd4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | tied | better | graph_prior_only | 40 | 4a189a72176a | 64f35244bcf0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | tied | better | graph_prior_only | 40 | c78791f9d9aa | eceb8f31eb03 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | tied | better | graph_prior_only | 40 | d9bdd81faea3 | 6854374441f2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | tied | better | graph_prior_only | 40 | 05bff8693f9c | a848f5276a7c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | tied | better | graph_prior_only | 40 | 75bfb725e244 | 1d625ce169eb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | tied | better | graph_prior_only | 40 | 9bf7555aeacb | aba824ea26a1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | tied | better | graph_prior_only | 40 | 337690b9c9eb | 1efd9ad50157 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | tied | better | graph_prior_only | 40 | 7ea5b0fe5898 | cedd7d9a0893 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | tied | better | graph_prior_only | 40 | 3d5741bc7f15 | af3f7df12761 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | tied | better | graph_prior_only | 40 | 30d113879d2d | 143888518c01 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | tied | better | graph_prior_only | 100 | e18137c972a3 | 1748625a14b7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | tied | better | graph_prior_only | 40 | 5dcf44fb33ef | 7d0a223520bd |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | tied | better | graph_prior_only | 40 | d87cf4dd79bd | d5e8669d0829 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | tied | better | graph_prior_only | 40 | 0cd35a7fe8e3 | 96fe9d44a0a3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | tied | better | graph_prior_only | 40 | 9aab3a4f139e | 5ca8d019336b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | tied | better | graph_prior_only | 40 | 3f027c000147 | 94ac1ca3cfa5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | tied | better | graph_prior_only | 40 | 7d7889b96a7c | 7d8bd12a9443 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | tied | better | graph_prior_only | 40 | 48aa7c8faba2 | 7aee5c71f280 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | tied | better | graph_prior_only | 40 | db960ffee6db | 1e6e1609a765 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | tied | better | graph_prior_only | 40 | dd707ca19ef0 | 372fcff96605 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | tied | better | graph_prior_only | 40 | d3b060468b3f | 51526d1c50ff |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | tied | better | graph_prior_only | 40 | cd4333b2e742 | 00edffd5d386 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | better | better | learned_route | 100 | f84dc11137b3 | 837903664ed6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | tied | better | graph_prior_only | 40 | 364296ea1b73 | 9bfc945359f8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | tied | better | graph_prior_only | 40 | c7dc09f4f37e | 6a95e0e55c18 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | tied | better | graph_prior_only | 40 | 4342aa2c769b | 85d30dde8f37 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | tied | better | graph_prior_only | 40 | d0dbd10f844f | 99d6c282250e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | tied | better | graph_prior_only | 40 | 1822b619e545 | 2e39fc7d8726 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | tied | better | graph_prior_only | 40 | 7d1472571d8a | 42119755b9bf |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | tied | better | graph_prior_only | 40 | b8c3230cde23 | 7f9d87d99431 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | tied | better | graph_prior_only | 40 | fcc35086c064 | 62d577cf3495 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | tied | better | graph_prior_only | 40 | a804c3efbebb | 73e8fa1c45db |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | tied | better | graph_prior_only | 40 | 42ddb475bbc2 | 82c2c89db9a3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | tied | better | graph_prior_only | 40 | aacec9e722aa | b0f167d1e7bd |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | tied | better | graph_prior_only | 40 | 6e2f536b4bc1 | 4f46f927b60c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | tied | better | graph_prior_only | 40 | c0b4a2378845 | 193d2bd3ad97 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | tied | better | graph_prior_only | 40 | 7d9ca1c47140 | ef39cb73ba31 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | tied | better | graph_prior_only | 40 | e5799ae0b28e | 1f781eb2879a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | tied | better | graph_prior_only | 40 | f6ca143550f4 | 4daef9907a3a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | tied | better | graph_prior_only | 40 | 360a892700a5 | 9d8a006807e8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | tied | better | graph_prior_only | 40 | ee3e65733fcd | f99990c18626 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | tied | better | graph_prior_only | 40 | 2d20fcfe8ab8 | c395f0cb503f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | tied | better | graph_prior_only | 40 | be1e1c6f9c81 | db420862008f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | tied | better | graph_prior_only | 40 | 55bd21567e7d | f37f0b6269e6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | tied | better | graph_prior_only | 40 | 99de50cd0aa9 | aa00cf238fcb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | tied | better | graph_prior_only | 40 | c023b1305718 | 43858d5fbe54 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | tied | better | graph_prior_only | 40 | 9c7b4bcb4046 | 19e8dd1a1fbb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | tied | better | graph_prior_only | 40 | b7731ea0e9f3 | 22ecd2d98c44 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | tied | better | graph_prior_only | 40 | b31ad3f0da26 | 58ec2f438c72 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | tied | better | graph_prior_only | 40 | 1a1f4976604b | 157a3a86cc10 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | tied | better | graph_prior_only | 40 | b625229336f1 | 740993a7cb13 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | tied | better | graph_prior_only | 40 | 2f83fcaa27a1 | f34b3fc61b23 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | tied | better | graph_prior_only | 40 | 7479f7fb5fe3 | 9f59a2be6a4c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | tied | better | graph_prior_only | 40 | b9ccda93b532 | 3cc54e53ee02 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | tied | better | graph_prior_only | 40 | 894fdab0d405 | f96d21c212c0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | tied | better | graph_prior_only | 40 | 8a538e3da812 | 04b6d0731fd8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | tied | better | graph_prior_only | 40 | b5525a642ed3 | b92b6449350c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | tied | better | graph_prior_only | 40 | e37a69006917 | 275d97bf56ee |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | tied | better | graph_prior_only | 40 | c09c29564f43 | 3b0f66c76eb5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | tied | better | graph_prior_only | 40 | 6585b9cc278d | b9d01f1b5bef |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | tied | better | graph_prior_only | 40 | 3fb9532757fc | 08615c252807 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | tied | better | graph_prior_only | 40 | f556dcbf7bc8 | 44590bbeb4e5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | tied | better | graph_prior_only | 40 | b106394cb0cc | 6cfbfccad4a5 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | tied | better | graph_prior_only | 40 | e2104a89a2b5 | acce829fce5a |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | tied | better | graph_prior_only | 40 | 20e7606e201b | 2900bb084a65 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | tied | better | graph_prior_only | 40 | f284d6ea6475 | d0141414ccc4 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | tied | better | graph_prior_only | 40 | a6e9b9fd0214 | 665673b79687 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | tied | better | graph_prior_only | 60 | e585a6159d0e | 840659bc2760 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | tied | better | graph_prior_only | 60 | 97305310a6d5 | 3f4ff559ee06 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | tied | better | graph_prior_only | 40 | 60ae8b8da37b | 7528d4f2eb21 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | tied | better | graph_prior_only | 40 | b19b620018d2 | 938f0a093f97 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | tied | better | graph_prior_only | 40 | 22056cbc6a16 | 1047144e35d9 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | tied | better | graph_prior_only | 40 | 0883e5999471 | e687f1cb445a |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | tied | better | graph_prior_only | 40 | 3f355daa6c7c | ac9c0d94d152 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | tied | better | graph_prior_only | 40 | a560c574b019 | 2442a58d1bbe |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | tied | better | graph_prior_only | 40 | 54ce422762e9 | 62c424769799 |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | tied | better | graph_prior_only | 40 | 8f28f3be95f3 | c07357b93ba3 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | tied | better | graph_prior_only | 60 | c061cc1449b9 | 003cdf5cde00 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | tied | better | graph_prior_only | 40 | 898d44346686 | 1b420ab1c732 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | tied | better | graph_prior_only | 40 | b3dd82c46efa | 8257b3ba2412 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | tied | better | graph_prior_only | 60 | 4e2ddcc1ab63 | b9513052675e |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | tied | better | graph_prior_only | 40 | 04418b713462 | 7edf22209bf2 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | tied | better | graph_prior_only | 40 | 9dac8498b2dc | e61846382672 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | tied | better | graph_prior_only | 40 | c5c40344adeb | 42eb5a948031 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | tied | better | graph_prior_only | 40 | c2c7fa612ddc | afc670e83aaa |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | tied | better | graph_prior_only | 40 | b2651e71a7a8 | 6539a03dd424 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | tied | better | graph_prior_only | 40 | 954c14e0350b | dc90becdfcfd |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | tied | better | graph_prior_only | 40 | 513503d11adc | 4d051be8fbbf |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | tied | better | graph_prior_only | 40 | 751b41b974ea | 04d3345e884d |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | tied | better | graph_prior_only | 40 | e61aa909f4fb | 890f88d04121 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | tied | better | graph_prior_only | 40 | 8551d615f620 | 6b0e71cda772 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | tied | better | graph_prior_only | 40 | 0d2d6df9ae09 | b74407b8e97e |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | tied | better | graph_prior_only | 40 | 01a731b2b92a | 316c8b29af50 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | tied | better | graph_prior_only | 40 | 98d396239c09 | 61885960a209 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | tied | better | graph_prior_only | 40 | 6b007fefee09 | 44a4289b482b |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | tied | better | graph_prior_only | 40 | beeac02e7896 | 5e6161ecf876 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | tied | better | graph_prior_only | 40 | 13a63dac28e0 | 3a08605601d6 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | tied | better | graph_prior_only | 40 | 1632c5dfdc8e | bc07e4e9cf20 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | tied | better | graph_prior_only | 40 | 26383a98153c | f39177871500 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | tied | better | graph_prior_only | 40 | ea870e72cf32 | 957beca3c5ab |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | tied | better | graph_prior_only | 40 | 652480aef008 | 23807bbb8762 |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | better | better | learned_route | 60 | f4ed8fdce933 | dc7776c27c74 |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | tied | better | graph_prior_only | 40 | b0fa69d4dcf2 | 755dcefe41b5 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | tied | better | graph_prior_only | 40 | 6d08ebc290cf | 464f7bf231b3 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | better | better | learned_route | 70 | 784e8af10a7a | 9c993aac1067 |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | tied | better | graph_prior_only | 40 | e2b64d51714d | 0274da3a2a08 |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | tied | better | graph_prior_only | 60 | dfb8fe997809 | 654fa5dccbaf |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | tied | better | graph_prior_only | 40 | b028664793b7 | 8ed2829a96e9 |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | tied | better | graph_prior_only | 40 | d28f5da4e30c | a9a14e892ed5 |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | tied | better | graph_prior_only | 40 | 3608a547ef28 | 90d7b2225d84 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | tied | better | graph_prior_only | 40 | 161378668c2a | ca1c84d52e36 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | tied | better | graph_prior_only | 40 | fdc91ec3bc72 | ce01203531af |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | tied | better | graph_prior_only | 40 | d49ad6af95bd | b4ed03b62bf0 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | tied | better | graph_prior_only | 40 | 84a34d1235fa | 8ca4d8ada297 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | tied | better | graph_prior_only | 40 | b2df6d8a3984 | 1e813fe37580 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | tied | better | graph_prior_only | 40 | b19411795fa8 | be99e2827657 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | tied | better | graph_prior_only | 40 | 2215483b7ae0 | 593992e93ff7 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | tied | better | graph_prior_only | 60 | fe085ffc7522 | 23c3c1e10acf |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | tied | better | graph_prior_only | 60 | ecd37e22477a | c4d71cf0ee13 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | tied | better | graph_prior_only | 40 | 1b39cd01ab6b | 1b7002bfa6df |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | tied | better | graph_prior_only | 40 | 5ffd189f47f1 | 3272c4339417 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | tied | better | graph_prior_only | 40 | eb0e8e5e9c7a | d944ec4c01dd |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | tied | better | graph_prior_only | 40 | a2901f4bd5c6 | 15be9448e458 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | tied | better | graph_prior_only | 40 | d153cfe52b9b | c47e165f039e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | 8e0b65b3c94c | 4c7d23091f85 |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | tied | better | graph_prior_only | 40 | f80c9168c772 | 3aea31178f99 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | tied | better | graph_prior_only | 40 | 4d375631a16b | 18e330e61ded |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | tied | better | graph_prior_only | 40 | e7ec4656390f | 2716219e5130 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | tied | better | graph_prior_only | 40 | 6d61dd5def95 | fd86c5d5ff01 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | tied | better | graph_prior_only | 40 | 4a2437ca3dce | 4e4a21f5b2d3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | tied | better | graph_prior_only | 40 | d654701df407 | 2e04c8e9e3b0 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | tied | better | graph_prior_only | 40 | 370b71a2a112 | 23da17d9ced7 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | tied | better | graph_prior_only | 40 | c1f692bf0335 | bdb2dec9c72d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | tied | better | graph_prior_only | 40 | d585939cd346 | 96941ba34df4 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | tied | better | graph_prior_only | 40 | 52cc5214e627 | c940b9c28f46 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | tied | better | graph_prior_only | 40 | 316e1dfb953c | 9240bcce4c1b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | tied | better | graph_prior_only | 40 | c746884c0a5b | b82d398f6caa |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | tied | better | graph_prior_only | 40 | 4f22179a7329 | 503db094e3f6 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | tied | better | graph_prior_only | 40 | add29e737250 | 7594eb3f0b00 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | tied | better | graph_prior_only | 40 | 2eeaca0f6080 | 2416d49b6214 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | tied | better | graph_prior_only | 40 | ba0fb47f2093 | 5d0fe42f192c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | tied | better | graph_prior_only | 40 | d7b23babf247 | 869d2659e584 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | tied | better | graph_prior_only | 40 | 40762f8040cd | 04a383631cca |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | tied | better | graph_prior_only | 40 | 81b30a85e542 | 671dc17abb67 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | tied | better | graph_prior_only | 60 | c444ec23b3e1 | 59a0400cf09b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | tied | better | graph_prior_only | 40 | 357c7bcb5c11 | b5380251dbbf |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | tied | better | graph_prior_only | 40 | e242d44d8c58 | 46db2aaf17c8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | tied | better | graph_prior_only | 40 | e30be677fece | ab49ecc694e6 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | tied | better | graph_prior_only | 40 | 554198cd2281 | 81cb38be57af |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | tied | better | graph_prior_only | 40 | 0e53fb93b343 | 2802adb5f2dc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | tied | better | graph_prior_only | 40 | ef49f0327205 | dd619cae26a8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | tied | better | graph_prior_only | 40 | 7fb4c49e7e26 | 9a83e62b5748 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | tied | better | graph_prior_only | 40 | 3022626fec4c | 8f5e81946397 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | tied | better | graph_prior_only | 40 | 2836ef51cdea | 32fe2a01e176 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | tied | better | graph_prior_only | 40 | 2ab418515628 | a1a3fb00afbe |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | tied | better | graph_prior_only | 40 | e152170167b5 | 3094e35246fc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | tied | better | graph_prior_only | 40 | 5e4e4eb53861 | 8cecbe79fb91 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | tied | better | graph_prior_only | 80 | 2dc4763969b5 | ef7387082139 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | tied | better | graph_prior_only | 80 | ee2111a91ba1 | c6969c155b22 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | tied | better | graph_prior_only | 40 | 539fad38f54c | 5299bf38f4db |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | tied | better | graph_prior_only | 80 | 677f55c9602b | ffd52c6edd26 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | tied | better | graph_prior_only | 40 | fd8207d978a2 | 55cda7ba26f4 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | tied | better | graph_prior_only | 40 | 7f6dc261edb6 | b09551cac3c6 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | tied | better | graph_prior_only | 40 | 3bea89f10087 | 2d1ba379a5bf |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | tied | better | graph_prior_only | 40 | 2010dc527087 | 9494541477b0 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | tied | better | graph_prior_only | 40 | d952e0b2c2b8 | 95448f4c0111 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | tied | better | graph_prior_only | 40 | 530c91488664 | 7e260931e9ed |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | tied | better | graph_prior_only | 40 | 34532029e9af | 7a3f82d8657a |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | tied | better | graph_prior_only | 40 | ed84f1b1a2e1 | 5ad80e7c8d82 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | tied | better | graph_prior_only | 40 | 86dd409f6d65 | 41d02567010a |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | tied | better | graph_prior_only | 40 | c121d0accc6c | 0887d7835b84 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | tied | better | graph_prior_only | 40 | 9a958c4f65d0 | fa3f84862547 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | tied | better | graph_prior_only | 60 | 41e09e18c4a5 | 769ad5f81a6d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | tied | better | graph_prior_only | 40 | 872009e1ebad | b83315b3abce |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | tied | better | graph_prior_only | 60 | 50c0c15a77b3 | c0d4ae4059b0 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | tied | better | graph_prior_only | 40 | cd40f5d92ed3 | aad7b8c15ecc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | tied | better | graph_prior_only | 40 | 54f8c998e188 | 3b3eb1d07c12 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | tied | better | graph_prior_only | 70 | 6c26acefef29 | 96d3bb2f05e4 |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | tied | better | graph_prior_only | 40 | e429e5239297 | 90f5cfdea39e |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | tied | better | graph_prior_only | 40 | c460d4789a51 | b3fee9c2025b |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | tied | better | graph_prior_only | 40 | dd72e99b3e16 | c7a93547db14 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | tied | better | graph_prior_only | 40 | 41e020f30fad | 2d4d1c321da0 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | tied | better | graph_prior_only | 60 | f5b0beea0e56 | a189ef06f0d4 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | tied | better | graph_prior_only | 40 | 6b6b894bf1c6 | 9af4e522e158 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | tied | better | graph_prior_only | 40 | 025c12bfcf1b | 294b683d8cd5 |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | better | better | learned_route | 60 | 1c3f76ab74b5 | c8dafb277ccf |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | tied | better | graph_prior_only | 40 | 3fcc06b0326d | 7baf2ae77069 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | tied | better | graph_prior_only | 70 | 2e768115c8a0 | e6a8ab7a3f88 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | tied | better | graph_prior_only | 40 | 66b859e3bd31 | 3b6d49ac3448 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | tied | better | graph_prior_only | 40 | 3fa9f9ca4769 | bc40c45dc1bb |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | tied | better | graph_prior_only | 40 | 91e6f69182d4 | 5dd98ae037d8 |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | tied | better | graph_prior_only | 40 | 1246ae7669b6 | c991d61d83a1 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | tied | better | graph_prior_only | 40 | 524071a9d81b | 764c1e9c2e64 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | tied | better | graph_prior_only | 40 | 09389a6c6a12 | 67b9db855850 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | tied | better | graph_prior_only | 40 | f7248c61aa29 | 0fb1a3aa9869 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | tied | better | graph_prior_only | 40 | cb596245dddd | faa8b83cd22e |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | tied | better | graph_prior_only | 40 | 0a9209768475 | 30594ccb880d |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | tied | better | graph_prior_only | 40 | 6000552ab95e | 4c0f146cd9d3 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | tied | better | graph_prior_only | 40 | bdbec37609ec | 6eab78edb94c |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | tied | better | graph_prior_only | 70 | ff93bbe51d87 | 8b0d752d0fcf |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | tied | better | graph_prior_only | 40 | 58ef6d4dcfe6 | bb042a69313f |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | tied | better | graph_prior_only | 40 | 9d57571959f3 | d984a035699d |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | tied | better | graph_prior_only | 40 | 4320d93d939a | fca3718743d8 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | tied | better | graph_prior_only | 40 | b19ef204e6b7 | d8f6e6afbefd |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | tied | better | graph_prior_only | 40 | bc5170932c6b | b722a71b0711 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | tied | better | graph_prior_only | 40 | 4d2d1eab7eca | fd90769e55a9 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | tied | better | graph_prior_only | 40 | cf09436c310b | 4d9f8c4c8568 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | tied | better | graph_prior_only | 40 | 40b4acdc055d | fd861e3bd13d |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | tied | better | graph_prior_only | 40 | baa5ce54c521 | eb068aa35a3e |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | tied | better | graph_prior_only | 40 | 207311cb4f4e | f67eb44c3cc8 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | tied | better | graph_prior_only | 40 | caa83e3dc72a | bcb2bfd75bdb |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | tied | better | graph_prior_only | 40 | 3ce9e9efb19b | 1f039e943c38 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | tied | better | graph_prior_only | 40 | cb7d23fd2ff6 | 706bdba7e24a |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | tied | better | graph_prior_only | 40 | 4ae58862b193 | 1375abe790e3 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | tied | better | graph_prior_only | 40 | 0da0642d9492 | 510efd90a0cb |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | tied | better | graph_prior_only | 40 | 2471605f7afe | 384a1d5313f9 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | tied | better | graph_prior_only | 40 | 4f8643f58170 | 8ddff878d748 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | tied | better | graph_prior_only | 40 | 8026cd00432f | eece0b67c501 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | tied | better | graph_prior_only | 40 | d208d58e77b0 | 3a7e8457032f |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | tied | better | graph_prior_only | 40 | f5fd86d2dc85 | 482188603e71 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | tied | better | graph_prior_only | 40 | ec1abc256662 | 21f72807a96f |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | tied | better | graph_prior_only | 40 | 969df328415f | 33744ea61f50 |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | tied | better | graph_prior_only | 40 | 5e7b19751f33 | 30b442edeaff |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-735fa10d9b980e548807d9aa64644b33e42cabdfacc7bafc43a22fe349ef986e |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-18239e3deffbd43f5a954032c2cc45d2f262db842c7441c834adf3f7eac4c3ba |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-b14736d47443f34294dfa15ab68792c1ff39e90268b18a617922359cda384d4f |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-94fd77a364df7a49658c15c39a1224fbacfdab62a6f58b09949e24c03d8e3c2c |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-cf9189c2939d9a6eb757143170dd26f55d3b38e9008e6850b9af6768b943131e |
| worked-traces | worked-traces.md | none | sha256-1e032ecdf525a0579e43ed6a10bdb64225ef26f5f6e74b79056e041b6310fa6a |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-11d52aa52f02a0407624d83b5d1f80126075d59ece63b3c724c56bb68d19da28 |
