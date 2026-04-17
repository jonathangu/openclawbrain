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
- learned_route vs graph_prior_only (traces): 7 better, 396 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 403/403 (1)
- learned_route vs graph_prior_only (turns): 7 better, 396 tied, 0 worse
- regressions vs graph_prior_only: 0/403 (0)
- regressions vs no_brain floor: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 61/832 required-context phrases vs graph_prior_only 52/832
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- activation precision: explicit learned-route activation precision is 7/403 across 403 observed candidate turns
- activation precision proxy: selection-divergence proxy activation precision is 7/403 against graph_prior_only
- success-adjusted economics: learned_route used 392.285714 estimated prompt tokens, 0.000491 estimated prompt USD, and 10 ms serve-path latency per incremental win vs graph_prior_only 256.714286, 0.000321, and 8
- fail-open: observed 0/403 degraded learned_route turns in this replay lane

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 2 | 403 |
| graph_prior_only | 394 | 394 |
| learned_route | 7 | 401 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | tied | better | graph_prior_only | 60 | 6fd702c0120c | edad5498eb9f |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | tied | better | graph_prior_only | 40 | 55f47bf4f75d | ca383356bb66 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | tied | better | graph_prior_only | 70 | 198f0088b710 | a76d4c440d71 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | tied | better | graph_prior_only | 80 | 448a8ba8ad48 | a8f8e1338445 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | tied | better | graph_prior_only | 40 | e4089cbe0603 | a1fa7ad0c8c3 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | tied | better | graph_prior_only | 60 | e7f2433379b3 | b0cabba356f4 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | tied | better | graph_prior_only | 60 | f87f894d1c4a | 465339900a50 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | tied | better | graph_prior_only | 60 | f335f8d6f4d5 | 01628292f3b0 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | tied | better | graph_prior_only | 60 | aa4951f8f7c6 | d79a2fd2fd31 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | tied | better | graph_prior_only | 40 | 11b71e1d1cd4 | 767359f309ee |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | tied | better | graph_prior_only | 60 | c4f575391cea | a2a8ca1c2062 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | tied | better | graph_prior_only | 40 | 4eec72058403 | bd69b2774f03 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | tied | better | graph_prior_only | 40 | 057354448f25 | 57eb38169c93 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | tied | better | graph_prior_only | 40 | ae30bfc126f8 | d4d44491467d |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | tied | better | graph_prior_only | 40 | ac0f70288a3a | 0c9d81f2b7cb |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | tied | better | graph_prior_only | 40 | 2f8a2df39065 | 49eed6de580e |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | tied | better | graph_prior_only | 60 | 629d39667cc2 | f5de585541c6 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | tied | better | graph_prior_only | 40 | ba2eff71d240 | 9b702b0880ce |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | tied | better | graph_prior_only | 40 | e686632ad86b | 4770fcf46908 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | tied | better | graph_prior_only | 40 | 24d91b8be748 | a77febc2a23f |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | tied | better | graph_prior_only | 40 | b7e5d2b8d2ad | 26931654b9e9 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | tied | better | graph_prior_only | 100 | 89246c70e00e | f7f962093cca |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | tied | better | graph_prior_only | 40 | 8d392c625025 | d6fe992ce25e |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | tied | better | graph_prior_only | 40 | 1970840abb62 | ed2eb58dda3c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | tied | better | graph_prior_only | 100 | d4e4d6e70fdc | 74a5332f4ba3 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | tied | better | graph_prior_only | 40 | 423f8a53c36e | e7a8f6c6cca9 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | tied | better | graph_prior_only | 40 | 625995a436c0 | a8e7143c2263 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | tied | better | graph_prior_only | 100 | 920b28e6a504 | 6f0f5a4e2bc6 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | tied | better | graph_prior_only | 40 | 480e739c0229 | 3f2dccd8c5b5 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | tied | better | graph_prior_only | 40 | 712a5016b29f | 4d37e5cde8aa |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | tied | better | graph_prior_only | 40 | 14c2596ae990 | 76b91c2c0fcf |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | tied | better | graph_prior_only | 40 | 6de2616e7e08 | b967658363a9 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | tied | better | graph_prior_only | 60 | 41b64dcab6d8 | abb491f6f633 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 5baa1a4a8580 | 06ed82bc4982 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | tied | better | graph_prior_only | 40 | 8c684c4800d9 | a0a3152dba6a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | tied | better | graph_prior_only | 40 | 33e8b3af8662 | 9c6eb9a9f323 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | tied | better | graph_prior_only | 40 | 6c981a0a84a4 | 8fcebf98d5c2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | tied | better | graph_prior_only | 40 | f756e8bd0d22 | 5080f4454907 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | tied | better | graph_prior_only | 40 | c44a2071b8b2 | 3ffe2bc074b6 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | tied | better | graph_prior_only | 40 | d31305eafeac | b4e8ca0398f2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | tied | better | graph_prior_only | 40 | d33a87e01116 | 84cc968bbaad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | tied | better | graph_prior_only | 40 | a4c4d81da8bd | 9eed19502b06 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | tied | better | graph_prior_only | 40 | 76d60e183205 | 3869501d75bc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | tied | better | graph_prior_only | 40 | 3b25af24a771 | 12b2d027e2d3 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | tied | better | graph_prior_only | 40 | 0e841672c994 | ca388d0aec96 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | tied | better | graph_prior_only | 40 | 678609a3a3a8 | 522a1b097477 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | tied | better | graph_prior_only | 40 | 77c79a00c203 | af07c2ecc786 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | tied | better | graph_prior_only | 40 | 64e7e55645b4 | 08bb1d565398 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | tied | better | graph_prior_only | 40 | ad033b7edd25 | e519a7a7b811 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | tied | better | graph_prior_only | 40 | ed8f6971137e | 4e9194910279 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | tied | better | graph_prior_only | 60 | 153200b3e683 | 96b705fa8599 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | tied | better | graph_prior_only | 40 | 689ae3136535 | 0ae86cff5aa4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | tied | better | graph_prior_only | 40 | 1b848f2f1d6f | e89d13e27085 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | tied | better | graph_prior_only | 60 | acac13ef2fb7 | 6c67f4e16b7e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | tied | better | graph_prior_only | 40 | bd5926a898d7 | 36e47d3ece62 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | tied | better | graph_prior_only | 40 | 363c8368749d | 220183859626 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | tied | better | graph_prior_only | 40 | c8afc97ded94 | ec516ad1fa28 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | tied | better | graph_prior_only | 40 | 928afe52041a | a06ea7ba51bb |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | tied | better | graph_prior_only | 40 | 9fc6ecdf6629 | 59837c234fe0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | tied | better | graph_prior_only | 40 | 2db36af4242d | bcb3ae37393b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | tied | better | graph_prior_only | 40 | a7f1e7a9e04b | 86a5a3182c34 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | tied | better | graph_prior_only | 40 | cd8a4d68be16 | 4c3ee149e11c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | tied | better | graph_prior_only | 40 | d50387bddb26 | 3907dcaa7402 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | tied | better | graph_prior_only | 40 | 75e4ef46efda | 72f88b4bdad9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | tied | better | graph_prior_only | 40 | 06fbd959096c | 20ac59c08652 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | tied | better | graph_prior_only | 40 | 580a3a7f4165 | 88d5399f0775 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | tied | better | graph_prior_only | 40 | 24ed2268e314 | fa3c6152ddf4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | tied | better | graph_prior_only | 40 | fb8356210aab | dbef0109aed8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | tied | better | graph_prior_only | 60 | aceb127c2968 | 1d8b282e7675 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | tied | better | graph_prior_only | 40 | 04952b07c4ea | 6424e6ea8e88 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | tied | better | graph_prior_only | 40 | 2e6ce6929deb | 2c4f4ddd0b09 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | tied | better | graph_prior_only | 40 | d8b246032572 | 6c8e0ad784d1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | tied | better | graph_prior_only | 40 | 256f45de2c16 | 04c68c703324 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | tied | better | graph_prior_only | 40 | 3d04fef9accf | b462154e801b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | tied | better | graph_prior_only | 40 | f29fb9b342dd | 39f491fc9842 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | tied | better | graph_prior_only | 40 | f080b0f1e971 | 396786ed78ce |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | tied | better | graph_prior_only | 40 | 65d0eefa6654 | 9debe3ac15b1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | tied | better | graph_prior_only | 60 | 26e736797889 | d2a7bc880b86 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | tied | better | graph_prior_only | 40 | bb815009ccd8 | 25da4ce903b5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | tied | better | graph_prior_only | 40 | 0f57898c504b | 780e67901873 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | tied | better | graph_prior_only | 40 | b2112cafd14d | ac12b76638a0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | tied | better | graph_prior_only | 40 | 015ee4badf9b | 29c6807faf7b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | tied | better | graph_prior_only | 40 | dfb13cb9f299 | 4c4099275c01 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | tied | better | graph_prior_only | 40 | dd4fb68a0a06 | 82a8ccc6d974 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | tied | better | graph_prior_only | 40 | 4c9ce962e759 | 6a2ed0ddc209 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | tied | better | graph_prior_only | 40 | 6e0871e1f25c | 5f97fde9f9bd |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | tied | better | graph_prior_only | 40 | 3baf6663f72b | b7019561acc9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | tied | better | graph_prior_only | 40 | 6d1d6a23fd2c | e732c17d1d88 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | tied | better | graph_prior_only | 40 | 4961fe4c894d | 269625b62b50 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | tied | better | graph_prior_only | 40 | 9454defe2481 | 7b22f88bb275 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | tied | better | graph_prior_only | 40 | 461d3199f3f8 | 3d174682d401 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | tied | better | graph_prior_only | 40 | 99935fae77e8 | d6ee8668a1e7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | tied | better | graph_prior_only | 40 | db8de5439a10 | 2f8e1c96c32b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | tied | better | graph_prior_only | 40 | ab7f3f4247e7 | 2e40cf56fac7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | tied | better | graph_prior_only | 70 | 575c81ab04d5 | 4b6547a46d43 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | tied | better | graph_prior_only | 40 | 69dfc991268c | 476ef78cc081 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | tied | better | graph_prior_only | 40 | eecb5f2c9726 | d8c3387afa87 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | tied | better | graph_prior_only | 40 | f20e0023ba6c | a0a35d02f09b |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | tied | better | graph_prior_only | 40 | 5e87efffc128 | 27105c8ab07c |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | tied | better | graph_prior_only | 40 | 98063cec8480 | f098b79a6fcd |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | tied | better | graph_prior_only | 40 | 92e1e9f1ea83 | fcb20c1f57c7 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | tied | better | graph_prior_only | 40 | 658897c927ac | f33c5af12f47 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | tied | better | graph_prior_only | 40 | 245cf1891b73 | db94f5fa997a |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | tied | better | graph_prior_only | 40 | 2d54619b6167 | 7b4f057e6b85 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | tied | better | graph_prior_only | 40 | 231073034aa6 | 8aae205ae0d5 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | tied | better | graph_prior_only | 40 | 31dd6ab85118 | ea70a2eb7daf |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | tied | better | graph_prior_only | 40 | d1e06f1073fd | 051400268df1 |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | tied | better | graph_prior_only | 40 | 89f2dbea8e09 | 81349507f4b0 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | tied | better | graph_prior_only | 40 | 75940a41f32b | afc548d1ff2b |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | tied | better | graph_prior_only | 40 | bdcfb019a308 | e48891cd17c3 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | tied | better | graph_prior_only | 60 | a0326717d6ef | 587e322a84ba |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | tied | better | graph_prior_only | 40 | 8396ee623679 | 40aff0ead9dd |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | tied | better | graph_prior_only | 40 | 4dc99d90a369 | 4c1bc849bef2 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | tied | better | graph_prior_only | 40 | f99ac875b0f7 | ea65c47ca021 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | tied | better | graph_prior_only | 100 | deca12d7badc | 169693c1b9b5 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | tied | better | graph_prior_only | 40 | c34882df2472 | bf56aac9a861 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | tied | better | graph_prior_only | 40 | b6caebc25b40 | 086396ae851b |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | tied | better | graph_prior_only | 40 | 0ddfae91197e | f085a7f23593 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | tied | better | graph_prior_only | 70 | a3daee2e7fe8 | 84d74b641101 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | tied | better | graph_prior_only | 40 | be054f7dfe50 | 1ecee8c91b98 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | tied | better | graph_prior_only | 40 | 2abf546568e7 | 1e0768b14153 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | tied | better | graph_prior_only | 40 | 62c23ec00476 | 9f862e6b1532 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | tied | better | graph_prior_only | 40 | 0daeee616544 | cbc1f7939e59 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | tied | better | graph_prior_only | 40 | f9a171c412f9 | af7d1c391880 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | tied | better | graph_prior_only | 40 | 7dd2c468ef5e | 88c54885e650 |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | tied | better | graph_prior_only | 40 | 90788506cb35 | db6f2e886298 |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | tied | better | graph_prior_only | 100 | bf5e91ae0d81 | 2082423c993f |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | tied | better | graph_prior_only | 40 | 8aef65ce12fb | f032fdabe0cd |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | tied | better | graph_prior_only | 40 | 8b086f4f5d7b | 72ad82775753 |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | tied | better | graph_prior_only | 40 | d50f4de9c4e2 | ec7e9a090676 |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | tied | better | graph_prior_only | 80 | 1d65816f9dc4 | 804b7f101ba2 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | 970e4164e04f | 1e2471fc74a8 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | tied | better | graph_prior_only | 40 | a2f48e320348 | 55418e0526f4 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | tied | better | vector_only | 60 | 102e8e64924a | 70b9287fe55d |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | tied | better | graph_prior_only | 40 | 0754d246995e | 334c31d4569a |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | tied | better | graph_prior_only | 40 | 045b83862684 | 1373918e4627 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | tied | better | graph_prior_only | 40 | 6ab0271824eb | babf0669c9a8 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | tied | better | graph_prior_only | 40 | 73f04a455780 | 5ac5fa8ac280 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | tied | better | graph_prior_only | 40 | 133f9a3117f8 | b113a8ac7be3 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | tied | better | graph_prior_only | 40 | b00eb6ba19a5 | 9234f2cc72e3 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | tied | better | graph_prior_only | 40 | 13ff7be149d4 | 3d057a936c69 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | tied | better | graph_prior_only | 40 | 9a36a1e00700 | 839d3b575b58 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | tied | better | graph_prior_only | 40 | 23e8aa3a4297 | bccc7eb28901 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | tied | better | graph_prior_only | 40 | 88d5fc805887 | 7383d47c718d |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | tied | better | graph_prior_only | 40 | bbed7d8e736f | e6d88a4356e1 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | tied | better | graph_prior_only | 40 | 87c577eb6241 | 71be8e12c062 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | tied | better | graph_prior_only | 40 | 5f3c3bed1793 | fddedf4510d6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | tied | better | graph_prior_only | 40 | 1a485d05e295 | d62bdd7e3a1a |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | tied | better | graph_prior_only | 40 | 2300783b9b3a | 5f4586e650e1 |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 48bac3ad75d1 | d634de1a85c5 |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | better | better | learned_route | 80 | 69e2f0daac6a | 41c83cc1b104 |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | tied | better | graph_prior_only | 60 | f8dd6ad0a6dc | 479f502820ff |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | tied | better | graph_prior_only | 40 | d0198a0946c9 | ae1429b1e937 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | tied | better | graph_prior_only | 40 | e1eb28a60f0c | d9575c2a71af |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | tied | better | graph_prior_only | 40 | a8ace80115a6 | a785d699362e |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | tied | better | graph_prior_only | 40 | d73e1a95ce83 | 9cdedcf52710 |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | tied | better | graph_prior_only | 40 | 1d5bc1c50218 | 78b3106ce438 |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | tied | better | graph_prior_only | 40 | ccce48e013f6 | 5cf9994e827d |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | tied | better | graph_prior_only | 60 | e66403b17576 | 2b12ff4f61b1 |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | tied | better | graph_prior_only | 40 | d7a0ae68c9e5 | d9d7065c486b |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | better | better | learned_route | 100 | 04c0312980ab | a5c102cb4554 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | tied | better | vector_only | 70 | 3472b7b4be15 | 509045194619 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | tied | better | graph_prior_only | 40 | 7e4c67640810 | ce6233901a3a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | tied | better | graph_prior_only | 40 | 1ad0bea15c53 | 941d952963f1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | tied | better | graph_prior_only | 40 | b5553c758c66 | 9e8485729fa2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | tied | better | graph_prior_only | 40 | c0d8509e5b36 | ed1b30a87cd0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | tied | better | graph_prior_only | 40 | bcfd6a68a448 | a2f16b6b5cc7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | tied | better | graph_prior_only | 40 | e7d6f2fb221c | 6e84ced8a07e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | tied | better | graph_prior_only | 40 | 8cbeb506f95c | a183f46716e2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | tied | better | graph_prior_only | 40 | 37e115ee7d1a | 3109861a179f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | tied | better | graph_prior_only | 40 | 6e3bb73e3393 | 74d043d22d4e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | tied | better | graph_prior_only | 40 | fa7f57fb793f | e10337e2f940 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | tied | better | graph_prior_only | 40 | 42ace4741283 | 9aa9ec3263a2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | better | better | learned_route | 100 | 108afb632969 | 192ff65c4648 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | tied | better | graph_prior_only | 100 | a8192a1b6de6 | e59a970753a3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | tied | better | graph_prior_only | 40 | 00100089aa9a | 3742d8707428 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | tied | better | graph_prior_only | 40 | f48c10f59c17 | e2f522ab849b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | tied | better | graph_prior_only | 40 | bf8039bcb9a4 | 9e9682fa1397 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | tied | better | graph_prior_only | 40 | 91c95fa5a266 | 9aa825faecd9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | tied | better | graph_prior_only | 40 | 5105707b3550 | 70dff892d78d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | tied | better | graph_prior_only | 40 | c843b2dcf5e3 | 7b18d5ed744e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | tied | better | graph_prior_only | 40 | 2cdb06a4eef8 | 98fbebae018b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | tied | better | graph_prior_only | 40 | e2c29ba11f33 | 658e641a149c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | tied | better | graph_prior_only | 40 | db0c083b0fe3 | 727fd2b09987 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | tied | better | graph_prior_only | 40 | 7178d7d229ff | ba51c5eb7a65 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | tied | better | graph_prior_only | 40 | a54ffd858ba3 | 13ff50478124 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | tied | better | graph_prior_only | 40 | 69bb037db5af | 984e9bcfd550 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | tied | better | graph_prior_only | 40 | c73e5edf0316 | b92d4942a3ee |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | tied | better | graph_prior_only | 40 | 3083fc9fb9c0 | 2cf7ff534148 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | tied | better | graph_prior_only | 40 | 53cb40d8030f | d76126f369f0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | tied | better | graph_prior_only | 100 | 5e3f20757f45 | 404eba733f59 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | tied | better | graph_prior_only | 40 | a608be7c5a43 | 7ce69506c308 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | tied | better | graph_prior_only | 40 | bcb9d74f8352 | 761ce2b58452 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | tied | better | graph_prior_only | 40 | 4d1e3fa44742 | bb2a1a17df3e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | tied | better | graph_prior_only | 40 | 038b6a5c691b | c4599a5d08a0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | tied | better | graph_prior_only | 40 | 3c05d163890e | 06aedd1224b3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | tied | better | graph_prior_only | 40 | 16ac6062756e | b7e3e1aa83be |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | tied | better | graph_prior_only | 40 | e23311b77a58 | db565317f6dc |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | tied | better | graph_prior_only | 40 | 2273f36473aa | 01cb1214e0f6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | tied | better | graph_prior_only | 40 | f9ac05c18e64 | a04907ebea8a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | tied | better | graph_prior_only | 40 | e4408a344d1c | 06d3d6078f75 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | tied | better | graph_prior_only | 40 | a30bdf6aba8d | d373da54e373 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | better | better | learned_route | 100 | 9387f03245a6 | 3f26725744a7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | tied | better | graph_prior_only | 40 | 23c489116c03 | b15f702ed078 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | tied | better | graph_prior_only | 40 | 7f5fb203c2da | 9f344800fd22 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | tied | better | graph_prior_only | 40 | 344b45431973 | 5e4be0fcf5dc |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | tied | better | graph_prior_only | 40 | 42b89cdb71fa | d53728b8b82a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | tied | better | graph_prior_only | 40 | a4bfc5588ef0 | 0b71948e8332 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | tied | better | graph_prior_only | 40 | d58374a87648 | 41da976785fb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | tied | better | graph_prior_only | 40 | 7893b809df41 | a8f2221cb387 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | tied | better | graph_prior_only | 40 | 612a59f0d37a | fd513090afb2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | tied | better | graph_prior_only | 40 | 86674ad80075 | e71c31fc581b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | tied | better | graph_prior_only | 40 | 3f5cecb5d393 | 8e981659af62 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | tied | better | graph_prior_only | 40 | f9f99dcca4e5 | f265821eb99d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | tied | better | graph_prior_only | 40 | 5ecaf6c71ea6 | e869c48e4afc |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | tied | better | graph_prior_only | 40 | ff270cb57405 | cbf12339a5bf |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | tied | better | graph_prior_only | 40 | 7742ec56c9b7 | eee4e6c17388 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | tied | better | graph_prior_only | 40 | b9f204f38e20 | eed3b86ca2ba |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | tied | better | graph_prior_only | 40 | c34f413f4152 | 16d134183439 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | tied | better | graph_prior_only | 40 | 7f9c1b58f9a2 | 3650a6d787c6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | tied | better | graph_prior_only | 40 | 6a0336ca6db5 | fb551f8418ec |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | tied | better | graph_prior_only | 40 | aef596e5bbed | f9ecfa0e77b8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | tied | better | graph_prior_only | 40 | 5fc53a8709b2 | d959738002e1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | tied | better | graph_prior_only | 40 | ae85461f83d2 | 22fd390ab8d8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | tied | better | graph_prior_only | 40 | 4a3b2b1c30c6 | 426f8d264461 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | tied | better | graph_prior_only | 40 | 758dd38f2a14 | 23d7344fe5e7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | tied | better | graph_prior_only | 40 | 6dba59473e82 | d81ea80603f8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | tied | better | graph_prior_only | 40 | 540d8625cc9b | 89a805f3e16e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | tied | better | graph_prior_only | 40 | 85e383617d45 | 33bfee2a0832 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | tied | better | graph_prior_only | 40 | d0d177b51a7c | 19e9732e6fb7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | tied | better | graph_prior_only | 40 | 412817a7888a | b675e1b8f981 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | tied | better | graph_prior_only | 40 | 9d54a1c67550 | 28b03a8b6313 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | tied | better | graph_prior_only | 40 | 02d4cee4318e | ea8691992b37 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | tied | better | graph_prior_only | 40 | 1dab3eae893b | d740ee339a0e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | tied | better | graph_prior_only | 40 | 702b5bcd3971 | 81dcd1e5edf2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | tied | better | graph_prior_only | 40 | 881472d3be51 | 95a0a2d1cdc2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | tied | better | graph_prior_only | 40 | 8836a5dd3c71 | c57ca79f0a5e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | tied | better | graph_prior_only | 40 | 0c0c1f671e87 | 05e179ef4a6d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | tied | better | graph_prior_only | 40 | 244685bbb879 | 8f8ecc06a05a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | tied | better | graph_prior_only | 40 | 2d6b24bdad66 | 7ac46e6a2799 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | tied | better | graph_prior_only | 40 | e83ce5d6fd72 | e10653cb27f0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | tied | better | graph_prior_only | 40 | 44233c1800a0 | 0f2b8b8f4a15 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | tied | better | graph_prior_only | 40 | 3e369668c66c | 43256cb9c261 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | tied | better | graph_prior_only | 40 | 7024068e12c3 | 0ae8e8c26f07 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | tied | better | graph_prior_only | 40 | 962ca0ee1145 | 51c8a1b23612 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | tied | better | graph_prior_only | 40 | 18ef97c67937 | 6ed43ab4ce1a |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | tied | better | graph_prior_only | 40 | 7fae27573202 | 4da2fd43374f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | tied | better | graph_prior_only | 60 | a1ba5c653cb1 | 2a1e0d770323 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | tied | better | graph_prior_only | 60 | 6fd06c70b910 | 7741d4fce06f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | tied | better | graph_prior_only | 40 | f53506952e30 | f2cbbacfe117 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | tied | better | graph_prior_only | 40 | 9c7208a78358 | 6e425a8fc8d7 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | tied | better | graph_prior_only | 40 | 081e9663ee7d | 3a63bb78d918 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | tied | better | graph_prior_only | 40 | f382f9de7448 | 0830f2bc5d6c |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | tied | better | graph_prior_only | 40 | e0da4e682a33 | 7b49060c45fd |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | tied | better | graph_prior_only | 40 | 172fbd4af078 | ceabee34f55c |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | tied | better | graph_prior_only | 40 | 0c4193310498 | 15763e4bcfe7 |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | tied | better | graph_prior_only | 40 | 75a5f88859cb | 0c5616a2c216 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | tied | better | graph_prior_only | 60 | 6aaac3c480dc | c3ae2278477a |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | tied | better | graph_prior_only | 40 | cd2145a5118a | 110d2f24cb57 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | tied | better | graph_prior_only | 40 | 190be16eccca | d0ba31dc630b |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | tied | better | graph_prior_only | 60 | d3b931dbb60b | 4f109a9f5ee2 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | tied | better | graph_prior_only | 40 | 4e8952aba4cc | 8671ff45faf5 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | tied | better | graph_prior_only | 40 | 7833563810b1 | 8bab86248696 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | tied | better | graph_prior_only | 40 | 65e4f2347330 | 49165906215e |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | tied | better | graph_prior_only | 40 | 74e0f8830050 | 08305782058f |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | tied | better | graph_prior_only | 40 | 3a012d0c9109 | 8f83f993a6ce |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | tied | better | graph_prior_only | 40 | d8fc4015624e | b34138c4c30d |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | tied | better | graph_prior_only | 40 | 655dfd58b346 | 174f781c5638 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | tied | better | graph_prior_only | 40 | 5cda49d93182 | 0d5665c5f96d |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | tied | better | graph_prior_only | 40 | 08d9f8dc2481 | 33498031bc92 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | tied | better | graph_prior_only | 40 | 03568a8fd23d | 8ef4ea9ff1e5 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | tied | better | graph_prior_only | 40 | 8c1c6bfa6813 | 87e8b95773f1 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | tied | better | graph_prior_only | 40 | 0fcd37b3d7cf | bd32eabf4bc5 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | tied | better | graph_prior_only | 40 | b1b0a3766556 | 1ce064ffb7af |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | tied | better | graph_prior_only | 40 | ecaab8534f90 | a989d4133495 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | tied | better | graph_prior_only | 40 | 4463e4e05da9 | e9bc507291ce |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | tied | better | graph_prior_only | 40 | ee5342dbbe26 | 865b1faa0f4c |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | tied | better | graph_prior_only | 40 | 350abaa2a17c | d0956dae3bed |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | tied | better | graph_prior_only | 40 | 898b4381099d | 9aafb9195cdc |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | tied | better | graph_prior_only | 40 | 2c305319e5c7 | 5a8c74b47c06 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | tied | better | graph_prior_only | 40 | 8dc7c9152eb0 | 17be8c6e741e |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | better | better | learned_route | 60 | 85ade81fba8f | 69b3b44d92cc |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | tied | better | graph_prior_only | 40 | 8a1665b0be52 | a7d2cc7edf43 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | tied | better | graph_prior_only | 40 | 02fa0694bc54 | 6f60e3d49161 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | better | better | learned_route | 70 | 09eba95216e3 | 0910a3ca8d3c |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | tied | better | graph_prior_only | 40 | ef691d785738 | 276fad747cd9 |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | tied | better | graph_prior_only | 60 | 1b67f231a566 | 092d111ea3c2 |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | tied | better | graph_prior_only | 40 | 192de5ae6c5c | cc5d9233badf |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | tied | better | graph_prior_only | 40 | b8e824a19859 | 3bfdf3e9e469 |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | tied | better | graph_prior_only | 40 | 08cbc9600727 | 17f0a55f563d |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | tied | better | graph_prior_only | 40 | 8ce3c707df67 | dde8e9da0e72 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | tied | better | graph_prior_only | 40 | 942cee35d50c | 2d9cb567cb21 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | tied | better | graph_prior_only | 40 | 4c524cf90b3c | 3ab9a9a39e29 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | tied | better | graph_prior_only | 40 | 76a254072f8f | 807240f8a599 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | tied | better | graph_prior_only | 40 | a0842ab7e1e2 | 6aa0ca6444a7 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | tied | better | graph_prior_only | 40 | 3d1f1dc381b1 | 67406e36cbcc |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | tied | better | graph_prior_only | 40 | 9fb89686cccb | ef6682a6e420 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | tied | better | graph_prior_only | 60 | 90e2c84ddc31 | e5ccc3d25058 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | tied | better | graph_prior_only | 60 | 2c8ae17ed6d9 | 558ef9b3474b |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | tied | better | graph_prior_only | 40 | ad00a654a276 | 2970b812a995 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | tied | better | graph_prior_only | 40 | 38da7f48784b | ba16caa1231b |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | tied | better | graph_prior_only | 40 | 96a3b8653249 | fc7212e6d912 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | tied | better | graph_prior_only | 40 | 4921d8304a2f | 66b7bd57ad85 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | tied | better | graph_prior_only | 40 | 39cc6f181db2 | 525a36be0b22 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | 49fdbbedadb4 | 54e06b697422 |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | tied | better | graph_prior_only | 40 | a61ce97093ca | d0d8e9085add |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | tied | better | graph_prior_only | 40 | ba2293b07cdd | 629bca159f37 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | tied | better | graph_prior_only | 40 | 174de60f7845 | 6ca7a7608bf0 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | tied | better | graph_prior_only | 40 | e758e9957e57 | 41736b837fca |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | tied | better | graph_prior_only | 40 | 8f092f26fdc7 | f3d27805fdac |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | tied | better | graph_prior_only | 40 | d8ed614d7c37 | 1fbd6eb4943e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | tied | better | graph_prior_only | 40 | 5419d26490f9 | 3c5dd292adc8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | tied | better | graph_prior_only | 40 | 065c483d2f8d | 4e06c4c5054e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | tied | better | graph_prior_only | 40 | 04bc404c3cfc | af86fcfba561 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | tied | better | graph_prior_only | 40 | cb435add5289 | 3bfd68bacf19 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | tied | better | graph_prior_only | 40 | dee9c264e1ca | 6beb382cd53b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | tied | better | graph_prior_only | 40 | d0c469ab1dd4 | ed53de3dc1d4 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | tied | better | graph_prior_only | 40 | 5c1cfe2a9828 | a095338e676f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | tied | better | graph_prior_only | 40 | 8480fa8d871c | 7dcc0008ee48 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | tied | better | graph_prior_only | 40 | 7b7e71b1406b | 950a75e71b04 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | tied | better | graph_prior_only | 40 | 433a956f3d1b | fa680a6b8925 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | tied | better | graph_prior_only | 40 | 89cc4e97fe4a | 339826197a6c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | tied | better | graph_prior_only | 40 | 6107f95aa9bd | e6e48d12c332 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | tied | better | graph_prior_only | 40 | b66c8bbcd677 | 9434969638be |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | tied | better | graph_prior_only | 60 | 6ef4c5dccb0e | ebef1d4a4674 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | tied | better | graph_prior_only | 40 | 0e5b53d4b440 | 92d66557a2b2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | tied | better | graph_prior_only | 40 | 07bbe8e336a8 | 3467d5a2fb7d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | tied | better | graph_prior_only | 40 | 84fbbf9a9681 | 853b5f877986 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | tied | better | graph_prior_only | 40 | a15362260aba | ff895f99e6c1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | tied | better | graph_prior_only | 40 | 21e80fcfeeac | 23f3b4b8ebd3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | tied | better | graph_prior_only | 40 | 00c0374509d5 | 807587f0e791 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | tied | better | graph_prior_only | 40 | e9db83adcdce | f6f985f74502 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | tied | better | graph_prior_only | 40 | 76fed809c531 | 09fe83d760c7 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | tied | better | graph_prior_only | 40 | 76f2a36c083d | 7064d0835649 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | tied | better | graph_prior_only | 40 | 73912fe10600 | d488e19a428f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | tied | better | graph_prior_only | 40 | 467389dc39f8 | 40f160b5df33 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | tied | better | graph_prior_only | 40 | ec9661f108d0 | 1401697e2686 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | tied | better | graph_prior_only | 80 | b2e172599c27 | 1e7a0178cc2d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | tied | better | graph_prior_only | 80 | 252ac8c090f4 | 78387f2c5e61 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | tied | better | graph_prior_only | 40 | d4ecaa700218 | 12df77b65190 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | tied | better | graph_prior_only | 80 | 97bf12739c33 | 7ddcc9e7ed15 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | tied | better | graph_prior_only | 40 | 96bc5feca8b2 | ae468d14cb8e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | tied | better | graph_prior_only | 40 | 7d6dd5c1c2b3 | e63db8435580 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | tied | better | graph_prior_only | 40 | fcabf45ab305 | fb5df18303f1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | tied | better | graph_prior_only | 40 | c044bd2ef5eb | 969fa07a4f06 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | tied | better | graph_prior_only | 40 | 27063084fada | 52710353cebb |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | tied | better | graph_prior_only | 40 | f0f512badc5b | 1f2cb1271393 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | tied | better | graph_prior_only | 40 | eee12890d73c | f0c326f7804f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | tied | better | graph_prior_only | 40 | 9bbde6e43ac2 | 785485e5a22f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | tied | better | graph_prior_only | 40 | 98b1acf30364 | 0a30b0f858b9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | tied | better | graph_prior_only | 40 | 6b6dc3eeee47 | 04877b061cf4 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | tied | better | graph_prior_only | 40 | 7db2c84bc622 | ddf4cb132d81 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | tied | better | graph_prior_only | 60 | fe872dc36dc6 | 1f5bc613ab6c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | tied | better | graph_prior_only | 40 | 589927bfb6c2 | 726dc417e585 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | tied | better | graph_prior_only | 60 | 8b232d6b03a4 | d660076ce111 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | tied | better | graph_prior_only | 40 | 1500957be1d5 | c5a833458793 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | tied | better | graph_prior_only | 40 | 18db4531f846 | b06587fb491d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | tied | better | graph_prior_only | 70 | 019197ef5db1 | 31e7228984db |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | tied | better | graph_prior_only | 40 | 1ab0d6ca86a3 | 15b8f8e5f3b1 |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | tied | better | graph_prior_only | 40 | 318062b90a15 | 062d6a8e25b5 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | tied | better | graph_prior_only | 40 | bdcd1ef6157f | 471765a6241b |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | tied | better | graph_prior_only | 40 | 15ead4dbaaff | 96b44db96003 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | tied | better | graph_prior_only | 60 | d230091f2cc5 | 6a3a3ac88deb |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | tied | better | graph_prior_only | 40 | 09c1d2aa22a3 | c73a882cc92a |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | tied | better | graph_prior_only | 40 | cd09734cd437 | be688a342d72 |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | better | better | learned_route | 60 | 3026b7ca105c | df96b20e38ae |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | tied | better | graph_prior_only | 40 | 8fe5034269a0 | b29d6e304627 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | tied | better | graph_prior_only | 70 | 9e313f0b0020 | 68788495cb2e |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | tied | better | graph_prior_only | 40 | ac48ee18db52 | 346a20e5a0fd |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | tied | better | graph_prior_only | 40 | 0ba3ce7dc96a | 0cb73e8dc850 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | tied | better | graph_prior_only | 40 | 3c6739495a90 | f259a102c49b |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | tied | better | graph_prior_only | 40 | 1414b1b87e98 | 8da460e3c820 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | tied | better | graph_prior_only | 40 | e3dc1b5671ac | 3f56734072cf |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | tied | better | graph_prior_only | 40 | fee0b8b7cf5b | 0f639c2ff6aa |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | tied | better | graph_prior_only | 40 | 92c89d874733 | 82a8ac606562 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | tied | better | graph_prior_only | 40 | 11c3e540b79f | d62af3c004a7 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | tied | better | graph_prior_only | 40 | f86533374adc | 2a4c63ff7304 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | tied | better | graph_prior_only | 40 | 1504c950def8 | 1d21eceb54ac |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | tied | better | graph_prior_only | 40 | 3507fec49012 | 038655649e53 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | tied | better | graph_prior_only | 70 | db4dd87b11d0 | 4ac4d8a10b8b |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | tied | better | graph_prior_only | 40 | fe11c1862e90 | 56c99b0a9a49 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | tied | better | graph_prior_only | 40 | c16f86461791 | 543835ce04c8 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | tied | better | graph_prior_only | 40 | 83c4af825496 | 6c702b0aa354 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | tied | better | graph_prior_only | 40 | 969dc9cecf83 | 536376415c26 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | tied | better | graph_prior_only | 40 | c9cf1104ceb0 | f0fa27aab191 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | tied | better | graph_prior_only | 40 | ca73e330925d | f9dae07a13e7 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | tied | better | graph_prior_only | 40 | 1f6dafcba837 | a3cbb206f86b |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | tied | better | graph_prior_only | 40 | a3514bf4ae90 | 682cff7065e2 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | tied | better | graph_prior_only | 40 | 3c7bf1a0abe5 | 045899702737 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | tied | better | graph_prior_only | 40 | 7602f137ff09 | b265e0924668 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | tied | better | graph_prior_only | 40 | a460963b5c17 | 57d43e608f08 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | tied | better | graph_prior_only | 40 | 59fd040a5923 | d775e0876c11 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | tied | better | graph_prior_only | 40 | ae61e7aa1049 | cf0890bf8204 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | tied | better | graph_prior_only | 40 | 81ed081dce21 | 01f102e027d0 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | tied | better | graph_prior_only | 40 | 722f1d3f1afb | 9631db8e6e56 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | tied | better | graph_prior_only | 40 | dcb2ae7e4129 | 8a8ceae17fca |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | tied | better | graph_prior_only | 40 | 1e10a25cc593 | 2fce89011dbd |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | tied | better | graph_prior_only | 40 | 291a424cdf55 | 4520d6dda05c |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | tied | better | graph_prior_only | 40 | b3990064848a | cb04c7e3ff59 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | tied | better | graph_prior_only | 40 | 91ebdfac5521 | 7d94b4b3e78e |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | tied | better | graph_prior_only | 40 | c07f11e0c164 | f84050d0626b |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | tied | better | graph_prior_only | 40 | c32c14663746 | 0b5e12b07030 |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | tied | better | graph_prior_only | 40 | a0f232328889 | d7637a39e902 |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-184ea3e5eafbea800cc0a42823164a0bb6335a8945e1a3848921bea5c76b4e38 |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-c436584e32859a8990944f9ba9507aee612335d11b05f1d793d8e6a0e5831b6f |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-bd2204348b7f9e3a653fba7a651b77fbba2b385609f96a3fc6d1c869c141fa5f |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-8326edf9a90fe8281e569f4390b1d8eb4d350415a7e9ad81f731dc7a40bbc809 |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-fdb423782a06aeeddc758f62736af3532d1e2d5b56243d7b64a64b0946c1c748 |
| worked-traces | worked-traces.md | none | sha256-0ab6b00601cf5c417e2efb07c9e6791a35bf7beb70be67a732209a0d5f1f6a1d |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-8b15350e953593893318675e9e96ed65f2938788ca8aec0bd28c0598576e1e09 |
