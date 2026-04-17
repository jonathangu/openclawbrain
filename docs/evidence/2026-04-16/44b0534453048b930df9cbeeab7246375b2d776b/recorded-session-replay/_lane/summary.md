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
- learned_route vs graph_prior_only (traces): 6 better, 397 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 403/403 (1)
- learned_route vs graph_prior_only (turns): 6 better, 397 tied, 0 worse
- regressions vs graph_prior_only: 0/403 (0)
- regressions vs no_brain floor: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 61/832 required-context phrases vs graph_prior_only 54/832
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- activation precision: explicit learned-route activation precision is 6/403 across 403 observed candidate turns
- activation precision proxy: selection-divergence proxy activation precision is 6/403 against graph_prior_only
- success-adjusted economics: learned_route used 393 estimated prompt tokens, 0.000491 estimated prompt USD, and 10 ms serve-path latency per incremental win vs graph_prior_only 253.666667, 0.000317, and 6
- fail-open: observed 0/403 degraded learned_route turns in this replay lane

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 2 | 403 |
| graph_prior_only | 395 | 395 |
| learned_route | 6 | 401 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | tied | better | graph_prior_only | 60 | 5e2e7200a9a9 | edad5498eb9f |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | tied | better | graph_prior_only | 40 | c32278228167 | ca383356bb66 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | tied | better | graph_prior_only | 70 | 75a14ec5fbf1 | a76d4c440d71 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | tied | better | graph_prior_only | 80 | db3ee089aa82 | 9be37e7fe4f2 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | tied | better | graph_prior_only | 40 | 270b3a0ae225 | 93a5defdba8f |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | tied | better | graph_prior_only | 60 | 7231df51a4d5 | 60b4d0a399e5 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | tied | better | graph_prior_only | 60 | e90bf4d5c191 | 4b81f8044d41 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | tied | better | graph_prior_only | 60 | 1544db054e9f | 73c36caa30c0 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | tied | better | graph_prior_only | 60 | f1de7ef850a6 | 9f10a863469c |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | tied | better | graph_prior_only | 40 | a45fb63c6036 | f13d68fa5e10 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | tied | better | graph_prior_only | 60 | ffecddb5ee66 | bf511b12968f |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | tied | better | graph_prior_only | 40 | bd6ebb282e1a | 338e0218c444 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | tied | better | graph_prior_only | 40 | 347767059473 | aa0126960148 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | tied | better | graph_prior_only | 40 | a3dc4a84a70a | f0ee407b20c5 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | tied | better | graph_prior_only | 40 | 43598475d878 | a601d092837c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | tied | better | graph_prior_only | 40 | e58cbcaf2207 | 14aed2475058 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | tied | better | graph_prior_only | 60 | 25f984fd5f05 | 3ade80b51c23 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | tied | better | graph_prior_only | 40 | fd63879e0bbf | a057eeb69ee0 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | tied | better | graph_prior_only | 40 | 23780c431752 | f7d079c8c6f7 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | tied | better | graph_prior_only | 40 | 69edd8266723 | d9e596bc0c32 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | tied | better | graph_prior_only | 40 | ba975c73d01f | 5390743af601 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | tied | better | graph_prior_only | 100 | a189c15a2879 | 4a145559c8f8 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | tied | better | graph_prior_only | 40 | df0673383542 | ac8038cc1906 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | tied | better | graph_prior_only | 40 | 54bcf737478c | b55d39d215c3 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | tied | better | graph_prior_only | 100 | 7b039d4ca9a1 | 299a75189dc8 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | tied | better | graph_prior_only | 40 | 1adb96d59f0d | ce8ef2228eeb |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | tied | better | graph_prior_only | 40 | 9b5b86a3b091 | 9a774fdde32a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | tied | better | graph_prior_only | 100 | 07ccc8608577 | 77ad94cdc284 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | tied | better | graph_prior_only | 40 | c12cd146c24f | 34b43e398d99 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | tied | better | graph_prior_only | 40 | 74a0f5ea786d | 4d4074b8588c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | tied | better | graph_prior_only | 40 | 915db855f9ad | ef420ae79a0a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | tied | better | graph_prior_only | 40 | 9d9ce380042a | 615f8255b5ae |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | tied | better | graph_prior_only | 60 | 4182c07a4dee | 60555b7cf4b9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 0f9957b409e1 | 06ed82bc4982 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | tied | better | graph_prior_only | 40 | f917064b227e | a0a3152dba6a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | tied | better | graph_prior_only | 40 | 456cba4293e7 | 9c6eb9a9f323 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | tied | better | graph_prior_only | 40 | eff321371f81 | 8fcebf98d5c2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | tied | better | graph_prior_only | 40 | 01a1f4645d16 | 5080f4454907 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | tied | better | graph_prior_only | 40 | aa9ee2fc47cf | 3ffe2bc074b6 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | tied | better | graph_prior_only | 40 | 0da9017ba2e8 | b4e8ca0398f2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | tied | better | graph_prior_only | 40 | baf99244499e | 84cc968bbaad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | tied | better | graph_prior_only | 40 | 7d3026d01dc0 | 9eed19502b06 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | tied | better | graph_prior_only | 40 | 911313a4bf98 | 3869501d75bc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | tied | better | graph_prior_only | 40 | 74c8797c1277 | 12b2d027e2d3 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | tied | better | graph_prior_only | 40 | e0dec1945d64 | ca388d0aec96 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | tied | better | graph_prior_only | 40 | e8df88291a6e | 522a1b097477 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | tied | better | graph_prior_only | 40 | 7f98c99d7137 | af07c2ecc786 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | tied | better | graph_prior_only | 40 | 3ddc0d54fa7b | 08bb1d565398 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | tied | better | graph_prior_only | 40 | 4b377da1058f | e519a7a7b811 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | tied | better | graph_prior_only | 40 | 65ef1d025eb8 | 4e9194910279 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | tied | better | graph_prior_only | 60 | ca6f5079a2c9 | 96b705fa8599 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | tied | better | graph_prior_only | 40 | 998177371205 | 0ae86cff5aa4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | tied | better | graph_prior_only | 40 | eaf502198ced | e89d13e27085 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | tied | better | graph_prior_only | 60 | 7eeced61f4db | 6c67f4e16b7e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | tied | better | graph_prior_only | 40 | f97c980b8362 | 36e47d3ece62 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | tied | better | graph_prior_only | 40 | b744d28650b6 | 220183859626 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | tied | better | graph_prior_only | 40 | f1eccb30a129 | ec516ad1fa28 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | tied | better | graph_prior_only | 40 | 29d85689ef14 | a06ea7ba51bb |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | tied | better | graph_prior_only | 40 | 02eabb6fe08d | 59837c234fe0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | tied | better | graph_prior_only | 40 | 981b0247f8eb | bcb3ae37393b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | tied | better | graph_prior_only | 40 | 7a7ffa12970e | 86a5a3182c34 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | tied | better | graph_prior_only | 40 | 589521f03e46 | 4c3ee149e11c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | tied | better | graph_prior_only | 40 | 5d2e8f865bcb | 3907dcaa7402 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | tied | better | graph_prior_only | 40 | 8ee98a2d89ff | 72f88b4bdad9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | tied | better | graph_prior_only | 40 | 94c7720a8dcf | 20ac59c08652 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | tied | better | graph_prior_only | 40 | 2d57ce3fba7b | 88d5399f0775 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | tied | better | graph_prior_only | 40 | 3e99027641bb | fa3c6152ddf4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | tied | better | graph_prior_only | 40 | b83103b725a8 | dbef0109aed8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | tied | better | graph_prior_only | 60 | f8a4634430e9 | 1d8b282e7675 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | tied | better | graph_prior_only | 40 | f2afce600aaa | 6424e6ea8e88 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | tied | better | graph_prior_only | 40 | 3ec17c779d9e | 2c4f4ddd0b09 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | tied | better | graph_prior_only | 40 | d7a836d00281 | 6c8e0ad784d1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | tied | better | graph_prior_only | 40 | a3e4e810da7d | 04c68c703324 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | tied | better | graph_prior_only | 40 | 2ff95c791544 | b462154e801b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | tied | better | graph_prior_only | 40 | dd93a65ffb8a | 39f491fc9842 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | tied | better | graph_prior_only | 40 | 17a702de326f | 396786ed78ce |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | tied | better | graph_prior_only | 40 | e9db2c9a120c | 9debe3ac15b1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | tied | better | graph_prior_only | 60 | ed4d9d436280 | d2a7bc880b86 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | tied | better | graph_prior_only | 40 | 757dfb32df08 | 25da4ce903b5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | tied | better | graph_prior_only | 40 | 73462e7ee085 | 780e67901873 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | tied | better | graph_prior_only | 40 | c5450d47cdc1 | ac12b76638a0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | tied | better | graph_prior_only | 40 | 283aafd46477 | 29c6807faf7b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | tied | better | graph_prior_only | 40 | 6ed0f9f7b0b6 | 4c4099275c01 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | tied | better | graph_prior_only | 40 | dfde89dc838d | 82a8ccc6d974 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | tied | better | graph_prior_only | 40 | 1e081fc93772 | 6a2ed0ddc209 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | tied | better | graph_prior_only | 40 | 928d37241d55 | 5f97fde9f9bd |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | tied | better | graph_prior_only | 40 | d438f153bafc | b7019561acc9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | tied | better | graph_prior_only | 40 | f9e8904222ed | e732c17d1d88 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | tied | better | graph_prior_only | 40 | 2a02f0802320 | 269625b62b50 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | tied | better | graph_prior_only | 40 | 323ee64808d9 | 7b22f88bb275 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | tied | better | graph_prior_only | 40 | 8dd808ce4364 | 3d174682d401 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | tied | better | graph_prior_only | 40 | 39724d56c654 | d6ee8668a1e7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | tied | better | graph_prior_only | 40 | b8ecec97ba6a | 2f8e1c96c32b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | tied | better | graph_prior_only | 40 | a9c8805ade9c | 2e40cf56fac7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | tied | better | graph_prior_only | 70 | 1ecd2dacb1b8 | 4b6547a46d43 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | tied | better | graph_prior_only | 40 | 0a0d68ad8fb8 | f91b5c96767d |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | tied | better | graph_prior_only | 40 | deb0f4ad5ff9 | b91bc04590ab |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | tied | better | graph_prior_only | 40 | ac343034dc41 | 1eab2cfb3e45 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | tied | better | graph_prior_only | 40 | d302791fd04c | 09a1ede3b4f2 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | tied | better | graph_prior_only | 40 | edc300915657 | 734485d13d63 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | tied | better | graph_prior_only | 40 | be78636cc92e | 50acb5a35689 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | tied | better | graph_prior_only | 40 | 0ca34101f67b | f66bcfdb676f |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | tied | better | graph_prior_only | 40 | f2ec24ae9cfc | cb41c71185ba |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | tied | better | graph_prior_only | 40 | 54ad65ef03ba | ffe97872c42a |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | tied | better | graph_prior_only | 40 | b295547520ef | 83c5d21a2a6a |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | tied | better | graph_prior_only | 40 | 03e4ef729a6e | 73d107e11b47 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | tied | better | graph_prior_only | 40 | 11445ee04917 | c7fa2bc25c28 |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | tied | better | graph_prior_only | 40 | 646e5bcc728a | 81349507f4b0 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | tied | better | graph_prior_only | 40 | 70f20fad46c7 | afc548d1ff2b |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | tied | better | graph_prior_only | 40 | 9ef65d1312f2 | e48891cd17c3 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | tied | better | graph_prior_only | 60 | bda965283a5d | 587e322a84ba |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | tied | better | graph_prior_only | 40 | 4eac2110408f | 40aff0ead9dd |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | tied | better | graph_prior_only | 40 | 8758be968890 | 4c1bc849bef2 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | tied | better | graph_prior_only | 40 | fbdf79fc65a3 | ea65c47ca021 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | tied | better | graph_prior_only | 100 | fc4699c487f3 | 169693c1b9b5 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | tied | better | graph_prior_only | 40 | edbb981cf2b2 | bf56aac9a861 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | tied | better | graph_prior_only | 40 | 41301df95522 | 086396ae851b |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | tied | better | graph_prior_only | 40 | 7e45b5cf585d | f085a7f23593 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | tied | better | graph_prior_only | 70 | 5a79d7ab6ea9 | 84d74b641101 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | tied | better | graph_prior_only | 40 | a72f3a2ad4c5 | 1ecee8c91b98 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | tied | better | graph_prior_only | 40 | ca551d541bc6 | 1e0768b14153 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | tied | better | graph_prior_only | 40 | de4009ee0635 | 9f862e6b1532 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | tied | better | graph_prior_only | 40 | 17f9969a0897 | cbc1f7939e59 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | tied | better | graph_prior_only | 40 | 91f2f6f14ba3 | af7d1c391880 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | tied | better | graph_prior_only | 40 | 6f11834b94bf | 88c54885e650 |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | tied | better | graph_prior_only | 40 | a70dee773e2f | db6f2e886298 |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | tied | better | graph_prior_only | 100 | 03657a7d1de0 | 37e08b64bbe3 |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | tied | better | graph_prior_only | 40 | 3f4487966453 | f032fdabe0cd |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | tied | better | graph_prior_only | 40 | f7f04e6a24c5 | 72ad82775753 |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | tied | better | graph_prior_only | 40 | 8ed7b6ea4965 | 8c9d7cbbd57b |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | tied | better | graph_prior_only | 80 | 951939f7d888 | 8adced6f7b2d |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | fc548a79b126 | 0eb12253298b |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | tied | better | graph_prior_only | 40 | 847e0a4538bd | 90eb74fd151d |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | tied | better | vector_only | 60 | d2f874f3e9c0 | 5381817d71b1 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | tied | better | graph_prior_only | 40 | 1b551b9792ce | f339cd3a2a34 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | tied | better | graph_prior_only | 40 | 3fa4d663fa08 | 3bdf3c2cfd1b |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | tied | better | graph_prior_only | 40 | 6abdfb1af773 | 5773e48e65b2 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | tied | better | graph_prior_only | 40 | 0bf56677440c | f758d3301dfd |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | tied | better | graph_prior_only | 40 | e837ea9b2d24 | 12f337774495 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | tied | better | graph_prior_only | 40 | ddb47f585737 | b49538d67d87 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | tied | better | graph_prior_only | 40 | f51955affffd | 40b2f6c66e0c |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | tied | better | graph_prior_only | 40 | 3fcaee971282 | 29038e0c9522 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | tied | better | graph_prior_only | 40 | dd95437ed5e9 | d21764d68917 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | tied | better | graph_prior_only | 40 | 4cca9febbc02 | a83a0b0c9afd |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | tied | better | graph_prior_only | 40 | fbfa962a6630 | 4aea0f9ec440 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | tied | better | graph_prior_only | 40 | 3dac52750a77 | 31d6aadd0434 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | tied | better | graph_prior_only | 40 | 194214fbb860 | 5f5b05710c9b |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | tied | better | graph_prior_only | 40 | cf412f3200cc | d59aa5afa8e9 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | tied | better | graph_prior_only | 40 | 068a1d08d626 | c056ade74b93 |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 597de5598f63 | 7453c0a5b363 |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | better | better | learned_route | 80 | 8e257e72d1dc | 0850fcdc9f0f |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | tied | better | graph_prior_only | 60 | 700553189ab8 | 2f8412ddcd7f |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | tied | better | graph_prior_only | 40 | c685f5a9a77f | 489254c87089 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | tied | better | graph_prior_only | 40 | 0b04794814e8 | 2e8bca19aaea |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | tied | better | graph_prior_only | 40 | 675ca5eeb68a | 9d7eed78d9b6 |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | tied | better | graph_prior_only | 40 | f5a0dbf6e7b2 | 41adb39d60d0 |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | tied | better | graph_prior_only | 40 | 232cc9bef000 | ad3a0602bdb9 |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | tied | better | graph_prior_only | 40 | 673b8ee4db69 | 4762abca8301 |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | tied | better | graph_prior_only | 60 | 0d050fd4c191 | d423bfd77b25 |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | tied | better | graph_prior_only | 40 | 6030bc0eda8d | d9d7065c486b |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | tied | better | graph_prior_only | 100 | e3fd3ffdd746 | f7eb00a0a0ed |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | tied | better | vector_only | 70 | 7d72b032412d | 95cd17cdfe32 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | tied | better | graph_prior_only | 40 | 17fd14f305ac | ae6462c32c92 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | tied | better | graph_prior_only | 40 | 154059aec0b5 | dc0fbeb352ef |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | tied | better | graph_prior_only | 40 | bef26eb1ff3f | a037c5b4e02e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | tied | better | graph_prior_only | 40 | 4c63872c605e | 168c51286516 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | tied | better | graph_prior_only | 40 | 9ce1288baec5 | d51f81f7f117 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | tied | better | graph_prior_only | 40 | 599c81371cb4 | d5b59879a4d0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | tied | better | graph_prior_only | 40 | ad28ead0cf5a | 9a78ab20bf66 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | tied | better | graph_prior_only | 40 | a14b7e540397 | 616d5d85e5b3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | tied | better | graph_prior_only | 40 | 90290fc119d5 | ff59854a3650 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | tied | better | graph_prior_only | 40 | f181ae83fc0a | 74ef290b0f38 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | tied | better | graph_prior_only | 40 | 191b05c67888 | a33d4bd39661 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | better | better | learned_route | 100 | b1d13458478a | 3d70f1808d9b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | tied | better | graph_prior_only | 100 | e9644919a290 | 83aab819e8ee |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | tied | better | graph_prior_only | 40 | 5cbeaaa12f46 | 55c402b2728f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | tied | better | graph_prior_only | 40 | 3b837e77f190 | 641e4a27870e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | tied | better | graph_prior_only | 40 | 10bedf453910 | ff8474487ebf |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | tied | better | graph_prior_only | 40 | b2ba89e05121 | 2b314b9369cf |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | tied | better | graph_prior_only | 40 | 8c08352d88dd | 1be9e9f185be |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | tied | better | graph_prior_only | 40 | c79264ca9411 | 3dc629451034 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | tied | better | graph_prior_only | 40 | b45fbf7cfe72 | 0ceaab84e7ba |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | tied | better | graph_prior_only | 40 | 2250d05681c9 | cfd46c757de6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | tied | better | graph_prior_only | 40 | eddd10db35e1 | 8e4cf4fa5276 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | tied | better | graph_prior_only | 40 | 6735f01e13fa | e1f89765e299 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | tied | better | graph_prior_only | 40 | 2e21d8aa6a57 | 366d8f61b462 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | tied | better | graph_prior_only | 40 | 20028703b074 | fd768482bc3b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | tied | better | graph_prior_only | 40 | 1032cf8fdd08 | f9fd980a7e8d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | tied | better | graph_prior_only | 40 | de503126f029 | 0b4543217565 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | tied | better | graph_prior_only | 40 | c63c2a861b80 | 88c322f4d12a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | tied | better | graph_prior_only | 100 | 21b2562dd571 | 626a2cb0dca9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | tied | better | graph_prior_only | 40 | b7d4b3bcd9f2 | 0e77e865431b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | tied | better | graph_prior_only | 40 | e805471f9b52 | c78ba9158ca3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | tied | better | graph_prior_only | 40 | ca87b969c5fc | 505c532aeef3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | tied | better | graph_prior_only | 40 | 287ba893a523 | 913a60ebd9c2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | tied | better | graph_prior_only | 40 | 8d145bb0054c | 5abbfce5405d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | tied | better | graph_prior_only | 40 | 2463504f2a2d | 330ee603b601 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | tied | better | graph_prior_only | 40 | 431bfe358d32 | a7bb78ab6fbb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | tied | better | graph_prior_only | 40 | 4a38ac4ac9b4 | 4a51e05729a1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | tied | better | graph_prior_only | 40 | 2e6c9e14505d | b3b7c3884fa7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | tied | better | graph_prior_only | 40 | 3574d344db5a | 18b512dee941 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | tied | better | graph_prior_only | 40 | f16e4afbfc85 | 1a71cc53da53 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | better | better | learned_route | 100 | 349437f0cfd9 | aefac928a4f7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | tied | better | graph_prior_only | 40 | f439417c47d5 | b683d0a2c0a7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | tied | better | graph_prior_only | 40 | 3e305b73542b | 9fe1bae85bab |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | tied | better | graph_prior_only | 40 | 43f76680a874 | 2580ffa25d59 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | tied | better | graph_prior_only | 40 | 3f8104295483 | 59779954175b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | tied | better | graph_prior_only | 40 | ce75d13f941b | 5c7fe4bdff10 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | tied | better | graph_prior_only | 40 | a935fa574a0d | 6550f7be5ef1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | tied | better | graph_prior_only | 40 | e574e8d85a12 | 652abb79b78e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | tied | better | graph_prior_only | 40 | 0f8c15771edc | 351a608bced0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | tied | better | graph_prior_only | 40 | b4100cd5178c | f581badac383 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | tied | better | graph_prior_only | 40 | eb224c2f4857 | e0525456990f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | tied | better | graph_prior_only | 40 | c7bfd8718cf0 | 706180bf72df |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | tied | better | graph_prior_only | 40 | 7b90307b8552 | 54eb048e1461 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | tied | better | graph_prior_only | 40 | ec66163ea5a9 | c584d2317ab8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | tied | better | graph_prior_only | 40 | 3caee8ccbacc | 1ca1b78230f8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | tied | better | graph_prior_only | 40 | 4217b1ad1c9c | 6b911c978484 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | tied | better | graph_prior_only | 40 | 6b208c32538d | 53a880b9b934 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | tied | better | graph_prior_only | 40 | 88243de33171 | b1eaab09b3c0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | tied | better | graph_prior_only | 40 | 7db04a0a48c7 | f37df52afd94 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | tied | better | graph_prior_only | 40 | 6d4df84012e9 | c6bafb7ff401 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | tied | better | graph_prior_only | 40 | d761a07ea014 | 2e29d31eb636 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | tied | better | graph_prior_only | 40 | 0d3978cd92f1 | 61ba1192c0d0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | tied | better | graph_prior_only | 40 | 56ad1c8cbadc | 82be90d056d0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | tied | better | graph_prior_only | 40 | ce29b51f8f5f | ea447f56ef0f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | tied | better | graph_prior_only | 40 | 4e33a8af505a | e83f5296ba3e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | tied | better | graph_prior_only | 40 | cd9002cafffd | 327bab9ec5c4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | tied | better | graph_prior_only | 40 | 1c5f548c8846 | 32419c937dd9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | tied | better | graph_prior_only | 40 | ccd98e20d086 | 5dd6f316c5b8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | tied | better | graph_prior_only | 40 | fc20c8e9983d | b4563b3222a6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | tied | better | graph_prior_only | 40 | 773c407d66a6 | 4e69f77fa550 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | tied | better | graph_prior_only | 40 | d6ef28f0561d | 1798cb09d69e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | tied | better | graph_prior_only | 40 | 0f4a12bd7964 | 503860611658 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | tied | better | graph_prior_only | 40 | 183af1919cb7 | 92215924756a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | tied | better | graph_prior_only | 40 | 2e6482cb3e9e | 2bd5d1bf5be0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | tied | better | graph_prior_only | 40 | 5af223dc0d57 | 2740846fb7fb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | tied | better | graph_prior_only | 40 | 9ff582e632b8 | 141ee5f525b3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | tied | better | graph_prior_only | 40 | b1831aef17b1 | b04b096de428 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | tied | better | graph_prior_only | 40 | 19ef54ab03cb | a79bbd260574 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | tied | better | graph_prior_only | 40 | 0f87ce14dd8d | a94ba9a987ad |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | tied | better | graph_prior_only | 40 | 70087a4c48f4 | 15bcbf9ebe79 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | tied | better | graph_prior_only | 40 | 97bbeeea6148 | 3550457154f1 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | tied | better | graph_prior_only | 40 | 798ca2107cbe | 0ae8e8c26f07 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | tied | better | graph_prior_only | 40 | 43c5695ba67f | 51c8a1b23612 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | tied | better | graph_prior_only | 40 | 23b42e8cd28e | 6ed43ab4ce1a |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | tied | better | graph_prior_only | 40 | 51af48f2d658 | 4da2fd43374f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | tied | better | graph_prior_only | 60 | 9b7db02ddcb7 | 2a1e0d770323 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | tied | better | graph_prior_only | 60 | cb09918608ff | 7741d4fce06f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | tied | better | graph_prior_only | 40 | 0d5ae2053a12 | f2cbbacfe117 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | tied | better | graph_prior_only | 40 | 9db5d9781473 | 6e425a8fc8d7 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | tied | better | graph_prior_only | 40 | 5ce53081eb11 | f5a08005bb30 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | tied | better | graph_prior_only | 40 | 732ba8cfeb79 | 44274d30357e |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | tied | better | graph_prior_only | 40 | f6c4513b1176 | 2f0103a5b9d8 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | tied | better | graph_prior_only | 40 | 6418d2b38c96 | 5fa8bcf7d036 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | tied | better | graph_prior_only | 40 | 229a3e52c292 | 20d812d90f0f |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | tied | better | graph_prior_only | 40 | 8820dd0c7fb4 | a83454a60bfb |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | tied | better | graph_prior_only | 60 | af3ca565c049 | 65fdf89e7d94 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | tied | better | graph_prior_only | 40 | 2019dc3d6ebf | c8f60dc0d05f |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | tied | better | graph_prior_only | 40 | ad25720b39cc | 6ed1bc144602 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | tied | better | graph_prior_only | 60 | 4d6fcfd8f0ce | 795d14bdb56f |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | tied | better | graph_prior_only | 40 | 0af00d1d1b02 | 9e3084b47700 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | tied | better | graph_prior_only | 40 | 6b5b4d38531d | 54a7504be69e |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | tied | better | graph_prior_only | 40 | b86f1b4d05ec | 554e914d2a86 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | tied | better | graph_prior_only | 40 | 249a72850cda | 832f490fc03a |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | tied | better | graph_prior_only | 40 | 72bdcaddebd5 | daece61a73e8 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | tied | better | graph_prior_only | 40 | 1663e587cf07 | 6af8bb85a1c3 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | tied | better | graph_prior_only | 40 | 9201e4dc715b | 7213c6e53327 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | tied | better | graph_prior_only | 40 | 77cf9c44f2f7 | 8c8a279d6f06 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | tied | better | graph_prior_only | 40 | 77c86fcc2f5c | 5fac10ba2985 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | tied | better | graph_prior_only | 40 | d0da583685a5 | c8694a3206fd |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | tied | better | graph_prior_only | 40 | 9eabd517a647 | df615a4858e0 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | tied | better | graph_prior_only | 40 | 3d7058ab8f8f | cba69aaea80f |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | tied | better | graph_prior_only | 40 | 38130db941f0 | 48b653a75b37 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | tied | better | graph_prior_only | 40 | 880bbed77f2a | 81f41d70c606 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | tied | better | graph_prior_only | 40 | f7f6625bc516 | ca96405f38f7 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | tied | better | graph_prior_only | 40 | ce11b25233c4 | 2c188e07e245 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | tied | better | graph_prior_only | 40 | de9a9b671884 | 702870c87d87 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | tied | better | graph_prior_only | 40 | 972d6b014a68 | 77947aee085f |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | tied | better | graph_prior_only | 40 | a3efe17806c2 | 4fe228463abe |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | tied | better | graph_prior_only | 40 | c4e29d930d41 | 0d5dd1f871ea |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | better | better | learned_route | 60 | db19b010f5ab | c63686cd7144 |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | tied | better | graph_prior_only | 40 | ba76c9aad395 | d28b539d2766 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | tied | better | graph_prior_only | 40 | db2f795036e9 | d609d2c40bed |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | better | better | learned_route | 70 | 1c8b0a3f19a8 | 158a24a0fe30 |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | tied | better | graph_prior_only | 40 | 142781cea318 | 23aebce26971 |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | tied | better | graph_prior_only | 60 | ea7ffccee8a7 | 09e5094cc786 |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | tied | better | graph_prior_only | 40 | 897b0a035db5 | 30e26bbfba9d |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | tied | better | graph_prior_only | 40 | 497034cd9b34 | 519eb4ea3fcf |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | tied | better | graph_prior_only | 40 | 3f989617db42 | 9b87ae171cd0 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | tied | better | graph_prior_only | 40 | ec4ff4bfe194 | 6ae31eb190e2 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | tied | better | graph_prior_only | 40 | 366030ca41e8 | 15d3f3746470 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | tied | better | graph_prior_only | 40 | 75932bba4a75 | a2dc6c22213e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | tied | better | graph_prior_only | 40 | eb6cfe4e974c | 9acc3b7c2bda |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | tied | better | graph_prior_only | 40 | f68e3316f2d1 | 9ab89733c232 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | tied | better | graph_prior_only | 40 | c396e53d89a3 | cd6582a111fc |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | tied | better | graph_prior_only | 40 | c4aa6f2055b1 | ecb5a371c572 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | tied | better | graph_prior_only | 60 | 37d1e5344c6c | 5b8a4a98c9a9 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | tied | better | graph_prior_only | 60 | 8fc7fbf3281f | ee5e18a81289 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | tied | better | graph_prior_only | 40 | 566b1255a6e5 | ac96ba8275af |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | tied | better | graph_prior_only | 40 | 807cc6b29882 | 690595dcf6d6 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | tied | better | graph_prior_only | 40 | 5567c3306422 | 117f95900ce5 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | tied | better | graph_prior_only | 40 | 4965db5d7701 | 22f7cadaf4b7 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | tied | better | graph_prior_only | 40 | 50bd08632bb2 | 6f4e3e56d166 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | 1093ca24a15c | 136d354c6f69 |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | tied | better | graph_prior_only | 40 | 7dde72e86dc4 | d1bfdff62f6d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | tied | better | graph_prior_only | 40 | 73e414855757 | e5616dd297b3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | tied | better | graph_prior_only | 40 | 00488b99e1aa | a296d4f7de89 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | tied | better | graph_prior_only | 40 | cb096af8d1f1 | efe559c52f95 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | tied | better | graph_prior_only | 40 | 03f3b0cfc13b | 0ed1f3981ceb |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | tied | better | graph_prior_only | 40 | 8b6d8547b914 | 4c29d02026b8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | tied | better | graph_prior_only | 40 | adb8c2713644 | 16ce557c2c23 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | tied | better | graph_prior_only | 40 | 371ea0c903de | 32e68e56cd66 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | tied | better | graph_prior_only | 40 | 03be0b9d4d6d | 983d03e338cb |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | tied | better | graph_prior_only | 40 | 1190bb5afdc9 | ad2f2d3e1df9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | tied | better | graph_prior_only | 40 | 66e66eedce2f | cb5c8627d909 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | tied | better | graph_prior_only | 40 | de0a81f6cdd7 | 6a1c1aceed40 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | tied | better | graph_prior_only | 40 | 53e2c48b870d | 27804a270e16 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | tied | better | graph_prior_only | 40 | d2821271ef36 | 28c6ae873480 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | tied | better | graph_prior_only | 40 | 4e18699d8bb7 | ab86415037e5 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | tied | better | graph_prior_only | 40 | c194156639ad | 5584e54d62ad |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | tied | better | graph_prior_only | 40 | acd846a0bdb0 | 1f7a1d7520b9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | tied | better | graph_prior_only | 40 | fdb8bf4459db | 625e8dbb42d2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | tied | better | graph_prior_only | 40 | 0a73555c5feb | 511a7b41684f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | tied | better | graph_prior_only | 60 | d2ec47a554be | 0338842912ba |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | tied | better | graph_prior_only | 40 | fa5536594824 | 2a47ab093c32 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | tied | better | graph_prior_only | 40 | 0c138a228294 | d29d8dbe4543 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | tied | better | graph_prior_only | 40 | cade37094601 | 548a52979e3e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | tied | better | graph_prior_only | 40 | 668fd9fdb30a | b326c83a4b8c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | tied | better | graph_prior_only | 40 | 70e5798c39f4 | bd5735247319 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | tied | better | graph_prior_only | 40 | 6022d719dbed | 8f3bbdbf0aa8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | tied | better | graph_prior_only | 40 | 6dcd17afc692 | 3decf5e2a83f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | tied | better | graph_prior_only | 40 | 0f99b9af3129 | 13f8b40885cd |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | tied | better | graph_prior_only | 40 | a5904b0de853 | ee0f208d062a |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | tied | better | graph_prior_only | 40 | 38df0aa21e3c | 1076dd760ae9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | tied | better | graph_prior_only | 40 | e89b0e780881 | 58fe96e5f1f0 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | tied | better | graph_prior_only | 40 | 2622cfe73fe9 | 410ef421fb63 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | tied | better | graph_prior_only | 80 | 8faa22427fe6 | fac30bf4a39d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | tied | better | graph_prior_only | 80 | 2a5aac9d217e | 161340ab3237 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | tied | better | graph_prior_only | 40 | 13a39fcbea42 | 7b4a2c0469db |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | tied | better | graph_prior_only | 80 | 39904e3fb5c3 | 54076c0db7c9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | tied | better | graph_prior_only | 40 | 4326ea59339e | 310bb3437a87 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | tied | better | graph_prior_only | 40 | 088c3e613e83 | a1f964b757aa |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | tied | better | graph_prior_only | 40 | 98edbed7bf7e | bba962f2d028 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | tied | better | graph_prior_only | 40 | 13851bc8c430 | 2d9c7de756b2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | tied | better | graph_prior_only | 40 | f2c23786a24e | 6a9758cf378f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | tied | better | graph_prior_only | 40 | ec77949ec672 | 29af38ee8ab8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | tied | better | graph_prior_only | 40 | 762caa448d2c | 88a4ed8181c4 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | tied | better | graph_prior_only | 40 | 4d5e6115fee3 | e08e71eb2e77 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | tied | better | graph_prior_only | 40 | 065c59a770f1 | 5d6960dcde9c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | tied | better | graph_prior_only | 40 | 402a9c4644cc | 76387b2ff66c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | tied | better | graph_prior_only | 40 | f33dde913e5c | d817bbe5616e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | tied | better | graph_prior_only | 60 | a13059daa738 | c7f870427124 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | tied | better | graph_prior_only | 40 | 922cf171b330 | 2ee40b3cd044 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | tied | better | graph_prior_only | 60 | 9595d08a512f | 6dd28507d777 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | tied | better | graph_prior_only | 40 | 0e727c7f4e7f | d74aabfc0d56 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | tied | better | graph_prior_only | 40 | 0679d275886c | ffac7ec6eaef |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | tied | better | graph_prior_only | 70 | c91cd3736c9f | 7bc6a59d97ac |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | tied | better | graph_prior_only | 40 | c5269b149fe7 | 205fdb0d0783 |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | tied | better | graph_prior_only | 40 | 16a5e5b1460d | 086be815d385 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | tied | better | graph_prior_only | 40 | 43d59b9c1a50 | 9bd24e18c59e |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | tied | better | graph_prior_only | 40 | 35f939b505ef | 9d4f31913ef0 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | tied | better | graph_prior_only | 60 | 8716251c21f8 | fac9826a0592 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | tied | better | graph_prior_only | 40 | 7fa8ba1f7a7f | 621e5837b1c7 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | tied | better | graph_prior_only | 40 | fe8e263e689c | 0c88dbf14458 |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | better | better | learned_route | 60 | d3cd30701bb6 | 61859e0fd12f |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | tied | better | graph_prior_only | 40 | 9319275e188f | 43df9b1947ce |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | tied | better | graph_prior_only | 70 | 6e2a71952b8e | bad4df646120 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | tied | better | graph_prior_only | 40 | 68069d26306d | 4a4724800c04 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | tied | better | graph_prior_only | 40 | 2c8c83a6f613 | e78377bc5c2d |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | tied | better | graph_prior_only | 40 | ee8b4d739898 | 9f6ebd737aba |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | tied | better | graph_prior_only | 40 | f6f84d3e218a | 89a587bc4210 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | tied | better | graph_prior_only | 40 | 574435977bb4 | f413fdc566e0 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | tied | better | graph_prior_only | 40 | fb8e6fa91c1f | 63f1f1001be3 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | tied | better | graph_prior_only | 40 | 8a3b19c87f82 | 2e512aefc10c |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | tied | better | graph_prior_only | 40 | df728a998acc | 53278a81416a |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | tied | better | graph_prior_only | 40 | d2b85ce9f781 | a8693fbc1d09 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | tied | better | graph_prior_only | 40 | 325db4d251ac | d83246b92b9e |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | tied | better | graph_prior_only | 40 | 6ff86611eca1 | dea60efd14f3 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | tied | better | graph_prior_only | 70 | 315095995a4a | 419270abab63 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | tied | better | graph_prior_only | 40 | c733e8181036 | fa528811dc70 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | tied | better | graph_prior_only | 40 | 4afa5edc3bdb | bfe4f55e0aa2 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | tied | better | graph_prior_only | 40 | c48b4f5f29ff | 7b9b1368c9f2 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | tied | better | graph_prior_only | 40 | 32f967d8c775 | 3a4b41d3c334 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | tied | better | graph_prior_only | 40 | de6cf2a67829 | dec7294f1793 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | tied | better | graph_prior_only | 40 | 64c76dfe0e4d | 2f8be160e9a6 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | tied | better | graph_prior_only | 40 | 637addd6e438 | dcb2f24b0469 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | tied | better | graph_prior_only | 40 | 91ebe2fc1954 | 20b842c35bd0 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | tied | better | graph_prior_only | 40 | c64002f21c44 | 0e828703ddca |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | tied | better | graph_prior_only | 40 | 56c3c6f2e239 | 8b6f1a3f973e |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | tied | better | graph_prior_only | 40 | c178e6fa21eb | 5f2a495bba1b |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | tied | better | graph_prior_only | 40 | fdcce6864a96 | 1d7470d65fdc |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | tied | better | graph_prior_only | 40 | 5a73efef3cc2 | c4d950898ac9 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | tied | better | graph_prior_only | 40 | a75d65cb76c7 | 2e88dd27d026 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | tied | better | graph_prior_only | 40 | 3514183ffc4e | cbcb5badf060 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | tied | better | graph_prior_only | 40 | b1dde2292399 | 96bf997541a1 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | tied | better | graph_prior_only | 40 | 89a46ee8c6a4 | d59b4ebaa8a8 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | tied | better | graph_prior_only | 40 | 0af14538fdfe | 6dbe3500bb74 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | tied | better | graph_prior_only | 40 | eb9d648add5e | 20eadf63648b |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | tied | better | graph_prior_only | 40 | b8046850b8c6 | 840d26a72874 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | tied | better | graph_prior_only | 40 | 6ecd39afd005 | ad17654d9cd8 |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | tied | better | graph_prior_only | 40 | dc482ad9c2d7 | 670d5372e3cb |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | tied | better | graph_prior_only | 40 | 88265a9945a3 | aa5a7a656bc3 |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-57abcb060b40838c722859568b1afbc1971bb299ae72be2f2f2dca02031f389d |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-5fd476bba1f35db15a394adf182a226825a4281db2d83deb82554288da11fb14 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-25466f78c239ade3cbb41bf871040970af4a983c997cfe02a224fb11aa5677fb |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-bd4b9dcdf5407deccab20555744a6f6208e61ac1a344d3a1bca5b3051f41dcf3 |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-5663d335243a40056e9f3d93655a933dd4ae9143677a86f36ad1fc97d05b4f98 |
| worked-traces | worked-traces.md | none | sha256-60eebd51dd8eff45fe50280662fe2a01943cfe37549fa144307a421b4042b933 |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-8bc7225e701b145460bd6b89de4ff371be37c83ac2a120c1805df1458ccabcff |
