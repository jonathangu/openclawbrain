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
- learned_route tie-or-better vs graph_prior_only (traces): 373/403 (0.925558)
- learned_route vs graph_prior_only (traces): 1 better, 372 tied, 30 worse
- learned_route tie-or-better vs graph_prior_only (turns): 373/403 (0.925558)
- learned_route vs graph_prior_only (turns): 1 better, 372 tied, 30 worse
- regressions vs graph_prior_only: 30/403 (0.074442)
- regressions vs no_brain floor: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 19/832 required-context phrases vs graph_prior_only 54/832
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- activation precision: explicit learned-route activation precision is 1/403 across 403 observed candidate turns
- activation precision proxy: selection-divergence proxy activation precision is 1/403 against graph_prior_only
- success-adjusted economics: learned_route used 169 estimated prompt tokens, 0.000211 estimated prompt USD, and 10 ms serve-path latency per incremental win vs graph_prior_only 293, 0.000366, and 5
- fail-open: observed 0/403 degraded learned_route turns in this replay lane

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 7 | 403 |
| graph_prior_only | 395 | 395 |
| learned_route | 1 | 366 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | worse | better | graph_prior_only | 60 | 35aea70768b6 | 15cf5d5481a7 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | tied | better | graph_prior_only | 40 | f894f64a3ac7 | 049df50f42f2 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | worse | better | graph_prior_only | 70 | 1a605e5def04 | e646d7e8876b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | worse | better | graph_prior_only | 80 | 6c289b77c9a7 | 6882305878ee |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | tied | better | graph_prior_only | 40 | 5a9a088c1b8a | cf3d2194c11e |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | tied | better | graph_prior_only | 60 | 9d524f8a613b | 9d7430f8bdef |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | worse | better | graph_prior_only | 60 | f77b0abc178a | 85f291112f20 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | worse | better | graph_prior_only | 60 | 2c72e7496528 | 6139c3db2b26 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | tied | better | graph_prior_only | 60 | 4d480792f201 | 74c2938c8674 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | tied | better | graph_prior_only | 40 | 0cd394fd0df0 | ae40af128be9 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | tied | better | graph_prior_only | 60 | 9ab815ff089d | d687b0b61532 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | tied | better | graph_prior_only | 40 | 575ea0839661 | 12f826a86e1b |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | tied | better | graph_prior_only | 40 | d34e1b0f9711 | 653b7c98c3e4 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | tied | better | graph_prior_only | 40 | 0f17f404d53f | 6a090893258b |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | tied | better | graph_prior_only | 40 | 63e8cde4c087 | 755c02df3e1c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | tied | better | graph_prior_only | 40 | 397fcbfd3dbc | bf0f32debb93 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | worse | better | graph_prior_only | 60 | dc0d8db4604b | c00c1c7d44eb |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | tied | better | graph_prior_only | 40 | 9aed3372ba7a | 6c33d17a18e3 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | tied | better | graph_prior_only | 40 | 9813c6532be1 | 45c5ad7855e1 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | tied | better | graph_prior_only | 40 | 6614aed85775 | dd48f603725d |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | tied | better | graph_prior_only | 40 | 5dbe3ddb1b82 | d0e19a954dcf |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | worse | better | graph_prior_only | 100 | 5d4b4cca613d | 6efd6460949a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | tied | better | graph_prior_only | 40 | 0c3aa727f691 | 8bf775685a75 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | tied | better | graph_prior_only | 40 | 7a5c4d2fbd36 | 6318dbebf21c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | worse | better | graph_prior_only | 100 | eb4109f85e74 | 5b8636e71511 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | tied | better | graph_prior_only | 40 | 1a2b9b2d5125 | 02829fd817ab |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | tied | better | graph_prior_only | 40 | e0f8e268f1f0 | 682fa4c3890d |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | worse | better | graph_prior_only | 100 | 9fc76d527410 | de08870bd4d9 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | tied | better | graph_prior_only | 40 | af9f7af86eee | 8e3cec64286b |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | tied | better | graph_prior_only | 40 | 4d739418d5e2 | 163646b94869 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | tied | better | graph_prior_only | 40 | ee497d3c5fc5 | 3ae8a0070df9 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | tied | better | graph_prior_only | 40 | ea2de097497c | dac53647db39 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | worse | better | graph_prior_only | 60 | f8076c5948cc | 18c46851afa4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | db421671ee00 | 90357b6a871a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | tied | better | graph_prior_only | 40 | f520f3c96f33 | f80d7020b4ab |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | tied | better | graph_prior_only | 40 | 14d44e14453b | 909d173ae2ae |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | tied | better | graph_prior_only | 40 | 99583554efa0 | 008d816e264e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | tied | better | graph_prior_only | 40 | 201c5844086c | 2a650637428e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | tied | better | graph_prior_only | 40 | 2adb368f70a3 | d1c6cc4b0a98 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | tied | better | graph_prior_only | 40 | e3c631c62ad9 | ba39769bf96b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | tied | better | graph_prior_only | 40 | 00f1386074f0 | e2e07f06af21 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | tied | better | graph_prior_only | 40 | 39fb001da76e | d19916ef8346 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | tied | better | graph_prior_only | 40 | e3d67d836990 | 24fdcf30b7d5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | tied | better | graph_prior_only | 40 | 4f13308d2b92 | e6168c244180 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | tied | better | graph_prior_only | 40 | 7864326d2b2b | a55b3a78233e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | tied | better | graph_prior_only | 40 | 8fe1aebec10d | b1c2a150c2bc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | tied | better | graph_prior_only | 40 | c37117a8c69f | 1cc92a2ab08d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | tied | better | graph_prior_only | 40 | 6ed6667bfaa6 | c8be769772ca |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | tied | better | graph_prior_only | 40 | d93c00c113d5 | 4a369615bec6 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | tied | better | graph_prior_only | 40 | 971c9f7fb74c | ed8bb07b3a7b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | tied | better | graph_prior_only | 60 | c74b1a41fe48 | 210f15ab0f6a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | tied | better | graph_prior_only | 40 | 00f76c617ccb | 88a61f07c248 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | tied | better | graph_prior_only | 40 | d0b28e685633 | d8c8701a2aec |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | worse | better | graph_prior_only | 60 | a7083426c37a | 267bb6d2b2d7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | tied | better | graph_prior_only | 40 | 3f99073c3724 | 048a503cc824 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | tied | better | graph_prior_only | 40 | 65859f43c231 | 0d4653d49f26 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | tied | better | graph_prior_only | 40 | c90375607636 | f67777629841 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | tied | better | graph_prior_only | 40 | 86f99db12301 | 6f358d8cb7f4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | tied | better | graph_prior_only | 40 | e41f48095d71 | 63428f7cb7ec |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | tied | better | graph_prior_only | 40 | 24f328991bb5 | ebecd77f0096 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | tied | better | graph_prior_only | 40 | 71ef43531b90 | b4352106fa73 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | tied | better | graph_prior_only | 40 | b60f455cd51b | ef2db2cd9a87 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | tied | better | graph_prior_only | 40 | 11e4c3991916 | 7bb7131d299e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | tied | better | graph_prior_only | 40 | 254c960abd8d | 04d85c3158e9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | tied | better | graph_prior_only | 40 | 960a5795ce08 | 13d6836106a9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | tied | better | graph_prior_only | 40 | 343f62122387 | 4c34cdb641ef |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | tied | better | graph_prior_only | 40 | d401782f3864 | b61fe118a1d2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | tied | better | graph_prior_only | 40 | 9fca5e64efd9 | 587d1ecb7b8d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | tied | better | graph_prior_only | 60 | 5bc6e50e8ed8 | a8eaafe0429e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | tied | better | graph_prior_only | 40 | 03c26e5879eb | 8eaa7fbcb4de |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | tied | better | graph_prior_only | 40 | cd16c342a07b | 5bf9b0d98d41 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | tied | better | graph_prior_only | 40 | c900039493c1 | f9497ae0ea47 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | tied | better | graph_prior_only | 40 | aa0c1c582c4a | 3761f78b6534 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | tied | better | graph_prior_only | 40 | 303ebc5fc754 | 03a1f1dceaad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | tied | better | graph_prior_only | 40 | 164f954d9f21 | d45870d79a2f |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | tied | better | graph_prior_only | 40 | 2e4f911deed3 | 38bdb46975d0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | tied | better | graph_prior_only | 40 | 49eb52615ba4 | 9f44831f263e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | worse | better | graph_prior_only | 60 | 5abafb6e57a1 | 3faccddaee2a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | tied | better | graph_prior_only | 40 | fa7536a22b88 | d3286363e67a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | tied | better | graph_prior_only | 40 | 019e9b49a11b | 5ba0b529152a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | tied | better | graph_prior_only | 40 | 7ad7a63ff7b7 | 8fd749aa6701 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | tied | better | graph_prior_only | 40 | b6231038d936 | df2b68c28abe |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | tied | better | graph_prior_only | 40 | 61cb3a1f3ef5 | c1e6137eb5c4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | tied | better | graph_prior_only | 40 | d39c1a90fa2c | 6412f1b8568a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | tied | better | graph_prior_only | 40 | cca7c32f4b1d | 999868909f3e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | tied | better | graph_prior_only | 40 | f2ab633d72d3 | 75669052ea9a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | tied | better | graph_prior_only | 40 | adebee663f1b | 9fbe0424afc5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | tied | better | graph_prior_only | 40 | 0701d24bc876 | 7b7b93d5cb65 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | tied | better | graph_prior_only | 40 | f18662b112f3 | 0817c00785eb |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | tied | better | graph_prior_only | 40 | cd1c21634a40 | cd23a2a8b9c1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | tied | better | graph_prior_only | 40 | 7791f03d22a1 | d8da044eb3f8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | tied | better | graph_prior_only | 40 | ce0a6b3f84ca | 51c7e05d1ed0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | tied | better | graph_prior_only | 40 | 36318d92862f | b218ade7730d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | tied | better | graph_prior_only | 40 | 0f79abb9f52a | b16de6f274d8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | tied | better | graph_prior_only | 70 | 038b1d41d706 | 1a9287223eab |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | tied | better | graph_prior_only | 40 | 104d93f35cba | 3f3cdd121184 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | tied | better | graph_prior_only | 40 | a954f80ca3f5 | 09b2f2fb74c3 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | tied | better | graph_prior_only | 40 | 5ba4039fe616 | 5d83692c90cb |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | tied | better | graph_prior_only | 40 | cda013d72c68 | e86b520fd57b |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | tied | better | graph_prior_only | 40 | a21a63813262 | ddc076b79805 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | tied | better | graph_prior_only | 40 | 9ec31f8c257d | 23e6b6a94c9e |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | tied | better | graph_prior_only | 40 | 4b6fbc2e9857 | d769d8137347 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | tied | better | graph_prior_only | 40 | 3f25acbc7942 | 0a23fb661f5c |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | tied | better | graph_prior_only | 40 | 9c9862c8a42b | dc9f9b5058b7 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | tied | better | graph_prior_only | 40 | df23c46f9f1b | 5d43f311142b |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | tied | better | graph_prior_only | 40 | 65253905fb60 | 69c9950e79c3 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | tied | better | graph_prior_only | 40 | 65b2817330e7 | 17d4b972a68e |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | tied | better | graph_prior_only | 40 | cc96d3e5c2ab | a0621fa057a3 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | tied | better | graph_prior_only | 40 | d77f19880ce8 | 9a3f17819272 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | tied | better | graph_prior_only | 40 | 55086aef84b8 | 0c1d2070c1d6 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | worse | better | graph_prior_only | 60 | 7402d0d7b2ff | c7830649c10d |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | tied | better | graph_prior_only | 40 | 93fcc47f3285 | 4e5ec02e6b68 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | tied | better | graph_prior_only | 40 | d9954cb86a9e | ec6e8b33b766 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | tied | better | graph_prior_only | 40 | e95120f3a9cd | 022caf7dbe13 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | worse | better | graph_prior_only | 100 | fc98d8c381ef | 1e0b9277137c |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | tied | better | graph_prior_only | 40 | bbf0e040a778 | 5677bb90cac8 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | tied | better | graph_prior_only | 40 | a8d348d3d18b | ba6a8213193c |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | tied | better | graph_prior_only | 40 | 51c15a953337 | a622c22298ec |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | tied | better | graph_prior_only | 70 | 8d02ee8c32fc | 4b9846f08891 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | tied | better | graph_prior_only | 40 | 1cce0a828c24 | 8268e8f590f9 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | tied | better | graph_prior_only | 40 | 4921ef1f7fd2 | daa33196cf1a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | tied | better | graph_prior_only | 40 | 01868ef4c270 | a3c4cc93084a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | tied | better | graph_prior_only | 40 | 9b1bfd58c06d | 4d5550987923 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | tied | better | graph_prior_only | 40 | e8140753d1bf | d2742938c213 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | tied | better | graph_prior_only | 40 | 67652a86f020 | 00a235e8a58f |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | tied | better | graph_prior_only | 40 | 8dc9eec6ecac | e206ba924576 |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | worse | better | graph_prior_only | 100 | 7fbe2017efb8 | ae91df3d52f0 |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | tied | better | graph_prior_only | 40 | ad136fd0a8d6 | 33513c341b71 |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | tied | better | graph_prior_only | 40 | 026704896def | db1459cae874 |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | tied | better | graph_prior_only | 40 | 90827f8c04f3 | a6cab82391cd |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | worse | better | graph_prior_only | 80 | fc3cf28abddf | e22a13f0a5ad |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | dec15a1d52fc | 4f95d224642f |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | tied | better | graph_prior_only | 40 | b9d9c0479265 | 924b6422d38e |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | tied | better | vector_only | 60 | 637c45451904 | 9291f05ae4fd |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | tied | better | graph_prior_only | 40 | 2d724a964284 | bf56d9123f42 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | tied | better | graph_prior_only | 40 | eb1f342010e8 | 841af0b97fe0 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | tied | better | graph_prior_only | 40 | 1a8b2fd57f53 | 3b7ef62345f4 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | tied | better | graph_prior_only | 40 | 1b69d8cd849a | 8bc125fb11b2 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | tied | better | graph_prior_only | 40 | 536f5db44aa5 | f1ec365244a8 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | tied | better | graph_prior_only | 40 | 5b8f23d4bb8b | 2f9011f3b741 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | tied | better | graph_prior_only | 40 | c710515823e1 | 10d1c2ae80ea |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | tied | better | graph_prior_only | 40 | 1b7e36b8a1e5 | cc5b2a14c9ad |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | tied | better | graph_prior_only | 40 | 9e69175f54fe | 913e872b6ac8 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | tied | better | graph_prior_only | 40 | 502970d7dd51 | fc6ebb6dd952 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | tied | better | graph_prior_only | 40 | 518df6d6b6da | a51101650f2a |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | tied | better | graph_prior_only | 40 | 0f42d381f95d | 4ca3e70e5177 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | tied | better | graph_prior_only | 40 | 4b48c5960bdc | 65cd8e0cafc6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | tied | better | graph_prior_only | 40 | 8fab0659b3fc | a25d3b97f2d6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | tied | better | graph_prior_only | 40 | 17e2cb0c318e | c698ecbe45c4 |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 1b87b1089cb9 | 294b691e2946 |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | tied | better | vector_only | 80 | a2e80a62303f | 6a65ea4faf62 |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | worse | better | graph_prior_only | 60 | 64f9ab97c08f | 684b0331dbf8 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | tied | better | graph_prior_only | 40 | 5951bcbd64e3 | b95406c85340 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | tied | better | graph_prior_only | 40 | f62abbf556a5 | d178b54dd66d |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | tied | better | graph_prior_only | 40 | bdee10809b09 | 1509784bccec |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | tied | better | graph_prior_only | 40 | ed8370dc6dd3 | 88f30af716b2 |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | tied | better | graph_prior_only | 40 | 4dbe009e42da | 2c7d920c1dd7 |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | tied | better | graph_prior_only | 40 | 9d60c411b0ef | e8a34b312b17 |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | worse | better | graph_prior_only | 60 | 845dd6930b96 | d0b1b75aa58a |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | tied | better | graph_prior_only | 40 | 5267a20d2b74 | 8eec058e861b |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | worse | better | graph_prior_only | 100 | c17a4709b860 | db225c375eb4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | tied | better | vector_only | 70 | 2c1e49953ce8 | 985c00a70ef4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | tied | better | graph_prior_only | 40 | 89e8170ac9cb | 37b8bd284ae8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | tied | better | graph_prior_only | 40 | 60a3ceae406d | 4abd7eb7038b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | tied | better | graph_prior_only | 40 | 09f16ba10275 | a6b4925a56f6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | tied | better | graph_prior_only | 40 | 505dc91c341f | e76b50a13db3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | tied | better | graph_prior_only | 40 | 6d4ac62c006f | e5525a064110 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | tied | better | graph_prior_only | 40 | a686019231f8 | 40625f2d3576 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | tied | better | graph_prior_only | 40 | 3b717b7e1ecf | 2b18dba2b0bf |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | tied | better | graph_prior_only | 40 | acacdab51fcb | 405c40d5b982 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | tied | better | graph_prior_only | 40 | 6cde272dfead | d8ea35af40df |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | tied | better | graph_prior_only | 40 | f9b1b23b0444 | 43b4a2f86bc7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | tied | better | graph_prior_only | 40 | b0ce987dc835 | c9bb0f0eb36a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | tied | better | vector_only | 100 | 78a469063522 | 1efdf2a64fbf |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | worse | better | graph_prior_only | 100 | 0b855bb1f75a | 45dfab8dfa8a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | tied | better | graph_prior_only | 40 | 70dcdfb7dc4e | 7848b40b4258 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | tied | better | graph_prior_only | 40 | 230ad4f883a4 | b5804a1dc2a6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | tied | better | graph_prior_only | 40 | 3bcb5f6d4afc | 7cc5cb65a7fa |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | tied | better | graph_prior_only | 40 | 3f0355f05f62 | 410cde793f26 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | tied | better | graph_prior_only | 40 | 30aa98f087ab | e23b263c8186 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | tied | better | graph_prior_only | 40 | d34c71893b4f | b61cf9cf134e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | tied | better | graph_prior_only | 40 | be07bfea6a7e | 623ed86967b6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | tied | better | graph_prior_only | 40 | 6b6c2bb88d45 | 5feddde7ab2d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | tied | better | graph_prior_only | 40 | 670259ee7e90 | ad4d83d5ea81 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | tied | better | graph_prior_only | 40 | 59a4d3f24122 | 8c0eceba1a6b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | tied | better | graph_prior_only | 40 | 32d4c2a93e53 | 35b97cb64667 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | tied | better | graph_prior_only | 40 | 09ff6a0afba2 | 32703d98c7d5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | tied | better | graph_prior_only | 40 | 6818c0eae848 | 11d4ebff01da |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | tied | better | graph_prior_only | 40 | bf137586d30b | 5c00a34f1e52 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | tied | better | graph_prior_only | 40 | b74999efcbff | 326e90888014 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | worse | better | graph_prior_only | 100 | 1a217e62cad3 | 5ff07f0578a7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | tied | better | graph_prior_only | 40 | 9d7b272424c8 | 2ac052300d3a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | tied | better | graph_prior_only | 40 | e957cfaf16ef | 062ffe1873de |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | tied | better | graph_prior_only | 40 | 623a760ee68e | 1a737ccf95f7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | tied | better | graph_prior_only | 40 | b629da4161b1 | c4f42018f311 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | tied | better | graph_prior_only | 40 | 5bec1ac88219 | 0f5f630ff54b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | tied | better | graph_prior_only | 40 | a40d2b86d91c | 69a90e650032 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | tied | better | graph_prior_only | 40 | 285bf1f76c09 | 1ca570eddee6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | tied | better | graph_prior_only | 40 | aee5f96ddd61 | cc6c04df23ad |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | tied | better | graph_prior_only | 40 | 5ef468fd6e26 | 59e191973dec |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | tied | better | graph_prior_only | 40 | 33855505a8ef | 49249f4d555f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | tied | better | graph_prior_only | 40 | e1c43544ff08 | fe69d8ae37df |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | tied | better | vector_only | 100 | 1c715d7a7995 | f902219c03d2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | tied | better | graph_prior_only | 40 | d74d50a67d1f | 36c106c0966c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | tied | better | graph_prior_only | 40 | ae8de367dc58 | cf414bccca90 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | tied | better | graph_prior_only | 40 | 712427121e68 | bb2bc3c371f4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | tied | better | graph_prior_only | 40 | be8c6cb970a2 | 3caa203bf5e2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | tied | better | graph_prior_only | 40 | 7e12842bdc9c | 61ca5698b5d4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | tied | better | graph_prior_only | 40 | 4b7c198db0eb | 8ce4ca4987ab |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | tied | better | graph_prior_only | 40 | 63b8e2cb39b1 | 29bc838552c5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | tied | better | graph_prior_only | 40 | 612d6a28d6c8 | 162070798301 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | tied | better | graph_prior_only | 40 | adea98cd8789 | 0d3d389e7ee8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | tied | better | graph_prior_only | 40 | 018e775e092d | 5a05f4ca5cc3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | tied | better | graph_prior_only | 40 | ab82d2ca6000 | dd2d35946ff5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | tied | better | graph_prior_only | 40 | 4fb48d4bc56a | e6042ec516ce |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | tied | better | graph_prior_only | 40 | daebc6ffeefe | 7637fcbeb33f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | tied | better | graph_prior_only | 40 | 760a6744a7fd | 3183aa262309 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | tied | better | graph_prior_only | 40 | 5d7822878c3e | 2a4eb17e4234 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | tied | better | graph_prior_only | 40 | 79abdc80aede | cfe3d57efe2d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | tied | better | graph_prior_only | 40 | 9cb11db452b9 | 3a637090bfe7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | tied | better | graph_prior_only | 40 | 3cbad720d591 | fb536228a65e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | tied | better | graph_prior_only | 40 | 5a2aa73d74f2 | 3faaa4e07e38 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | tied | better | graph_prior_only | 40 | 0fa8db83463e | 5ae16460a875 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | tied | better | graph_prior_only | 40 | ae06feb88fa3 | c0784fe59075 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | tied | better | graph_prior_only | 40 | 8f2efa5fb4c4 | 9eebbd854695 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | tied | better | graph_prior_only | 40 | 6eae098f4bf3 | 1a86f489e638 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | tied | better | graph_prior_only | 40 | 3546fa11ef06 | bf4c8fa7bfdc |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | tied | better | graph_prior_only | 40 | 162aa176a8df | be0c1359f127 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | tied | better | graph_prior_only | 40 | a923d8970d6d | e2819590c89e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | tied | better | graph_prior_only | 40 | da136a424d82 | 10a50dbbf027 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | tied | better | graph_prior_only | 40 | ca225147ef6b | 235bdacf642d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | tied | better | graph_prior_only | 40 | b295d722decf | 6a3c6f6f1737 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | tied | better | graph_prior_only | 40 | 8bd145dec037 | dfbe0fb6e2c0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | tied | better | graph_prior_only | 40 | 65576ca4fc01 | 91166a05bef1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | tied | better | graph_prior_only | 40 | 4ef8d3508a6f | 9e9a8d5cb744 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | tied | better | graph_prior_only | 40 | 004f5e613e26 | 6ce7b16a4e2a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | tied | better | graph_prior_only | 40 | ee84fddf69f5 | 0083a6534a9f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | tied | better | graph_prior_only | 40 | 076154eddbab | 8cb6aff5f0c8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | tied | better | graph_prior_only | 40 | 8a62b0fa1cfa | f84eddfa818e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | tied | better | graph_prior_only | 40 | 187405479401 | d5d3455fa77a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | tied | better | graph_prior_only | 40 | 5fad0001b224 | 0a8f53232c0f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | tied | better | graph_prior_only | 40 | 555b1f97574b | b42b92144a2e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | tied | better | graph_prior_only | 40 | 889db564d97a | 4954c609fdbc |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | tied | better | graph_prior_only | 40 | eeed2cb35bf8 | 6f16dff1b1ad |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | tied | better | graph_prior_only | 40 | 7ee04be6f821 | ddab66e349e7 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | tied | better | graph_prior_only | 40 | c8ab09513719 | 9edaa3b10a9f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | tied | better | graph_prior_only | 40 | 8c3251ea8176 | ebf44f1e0edf |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | worse | better | graph_prior_only | 60 | 2c79fd7493b6 | 1aa3f3c9b04c |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | tied | better | graph_prior_only | 60 | d861239bb005 | fe7f8b01f8fd |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | tied | better | graph_prior_only | 40 | bf55f7fc4dbf | 38bfb0530318 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | tied | better | graph_prior_only | 40 | 22a3679c5f2d | 015ed4563792 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | tied | better | graph_prior_only | 40 | 4566624cd09f | 606fdf4b7095 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | tied | better | graph_prior_only | 40 | 7741bc3943a7 | 81bacf59c718 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | tied | better | graph_prior_only | 40 | 9af4dc6304e3 | a641a914714e |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | tied | better | graph_prior_only | 40 | d307acb8741f | 70015b46aa97 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | tied | better | graph_prior_only | 40 | e0a0484248bf | 7452db4f9616 |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | tied | better | graph_prior_only | 40 | e141c7e74032 | 6b29f372c3d8 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | worse | better | graph_prior_only | 60 | b91a55e9081b | 01ab1151dc09 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | tied | better | graph_prior_only | 40 | 68281d6d320f | e9cab802330e |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | tied | better | graph_prior_only | 40 | 5bf0961d62e8 | 1f1c247aa1a3 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | worse | better | graph_prior_only | 60 | 19c9e0f88e83 | 480570809e1a |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | tied | better | graph_prior_only | 40 | 1013ff62b439 | 7372b7a22909 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | tied | better | graph_prior_only | 40 | e44ae3469d4c | cb2e4bc849c2 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | tied | better | graph_prior_only | 40 | 41a49fbdea4a | 0ae838ec684b |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | tied | better | graph_prior_only | 40 | e2492a9cfa44 | 66c7e4fbc253 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | tied | better | graph_prior_only | 40 | 30e58b030c47 | a710214392a8 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | tied | better | graph_prior_only | 40 | 4f4a9183fa4c | 6268d8b8ab3e |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | tied | better | graph_prior_only | 40 | 29700ad8e35f | cc27541342bc |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | tied | better | graph_prior_only | 40 | 1c230165e4aa | dba951b19959 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | tied | better | graph_prior_only | 40 | 044f145ead2d | 23de2cec299e |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | tied | better | graph_prior_only | 40 | 1a4b3a79d55f | 0d917c4665b0 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | tied | better | graph_prior_only | 40 | a160ed76dc4e | 917ba1c7a03b |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | tied | better | graph_prior_only | 40 | d1d4f03055de | e83107efffdd |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | tied | better | graph_prior_only | 40 | 2c615e13faea | 1b3fa8b467f3 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | tied | better | graph_prior_only | 40 | 6f050b5fde9a | 6deeeee9d60b |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | tied | better | graph_prior_only | 40 | 05294ee3d855 | 0854b5758ba2 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | tied | better | graph_prior_only | 40 | 9558a56c855b | 60906de34c61 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | tied | better | graph_prior_only | 40 | 9783d3d6901e | 0890fcb36e58 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | tied | better | graph_prior_only | 40 | b3a9cde73a6c | c76c6d85e3c5 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | tied | better | graph_prior_only | 40 | b05a37844855 | 628daa57e627 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | tied | better | graph_prior_only | 40 | 3d2af3192b43 | 47fad248d3ad |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | better | better | learned_route | 60 | 43007fb0a399 | bfe25e59c0db |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | tied | better | graph_prior_only | 40 | 9806c580e053 | b2fd830d5b71 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | tied | better | graph_prior_only | 40 | bf413d9389ea | bfba6a528305 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | tied | better | vector_only | 70 | 28dc436f929e | b60849552142 |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | tied | better | graph_prior_only | 40 | 91e7d4991b6f | 28ed5e90e255 |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | worse | better | graph_prior_only | 60 | 86638e89f9d3 | aac89ced61e6 |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | tied | better | graph_prior_only | 40 | edf0332495d5 | 7b1a4ef9ed19 |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | tied | better | graph_prior_only | 40 | 86fb474aa41f | 0acd6c33fa8d |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | tied | better | graph_prior_only | 40 | 3eb99bedd9ca | 4a504db5a191 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | tied | better | graph_prior_only | 40 | 34e1f2734002 | 2cc02f347f64 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | tied | better | graph_prior_only | 40 | 982a1822980d | 390d0cbf2d0a |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | tied | better | graph_prior_only | 40 | 37bda2274086 | 7f6ebb5c7fb3 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | tied | better | graph_prior_only | 40 | 4505cc98dfa0 | 1ee08a70b95e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | tied | better | graph_prior_only | 40 | 28993d855da6 | 932a4fccec49 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | tied | better | graph_prior_only | 40 | e3ca4f3b68a2 | 81bfa41fb8f5 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | tied | better | graph_prior_only | 40 | cf1b01abb36c | 44c6a26cba70 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | tied | better | graph_prior_only | 60 | 9219a317a33f | b7ed540fe910 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | tied | better | graph_prior_only | 60 | 805df70ab671 | d2513ea54cb0 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | tied | better | graph_prior_only | 40 | 5216232d086e | 9f343dfb67c4 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | tied | better | graph_prior_only | 40 | 20a7428d1641 | de2f6e891ec2 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | tied | better | graph_prior_only | 40 | 0654a330e1c4 | 04ca7aebd9b4 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | tied | better | graph_prior_only | 40 | 716a76accb73 | 0d8d58652e1e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | tied | better | graph_prior_only | 40 | 0ab456661922 | eef5a047776d |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | ddc33465a580 | be1c8309f4f8 |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | tied | better | graph_prior_only | 40 | 0b9f2d34bdf1 | 4a4fa6c26b47 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | tied | better | graph_prior_only | 40 | d12a0ce7a054 | c5e96e77a223 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | tied | better | graph_prior_only | 40 | d57b1f458771 | 6088e8eb318d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | tied | better | graph_prior_only | 40 | aae6a1366978 | eab0ce42eb52 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | tied | better | graph_prior_only | 40 | 730fbb5d65c7 | 69055d849a45 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | tied | better | graph_prior_only | 40 | bb1d4f40549c | 6610f9519a77 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | tied | better | graph_prior_only | 40 | fcae2bbf0fe5 | 933018960b17 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | tied | better | graph_prior_only | 40 | 244db6bee48d | d512475d235e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | tied | better | graph_prior_only | 40 | 1af501079765 | 7f80b49970b2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | tied | better | graph_prior_only | 40 | ab660ca7f7f3 | 6a64b276840e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | tied | better | graph_prior_only | 40 | 052b60d300d5 | 345e3590cb0f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | tied | better | graph_prior_only | 40 | ae47983bd3cf | ffde0ca47c66 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | tied | better | graph_prior_only | 40 | 199e143bb295 | 3c47ae691423 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | tied | better | graph_prior_only | 40 | 0ac0ae91512e | ae3e54ac93ab |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | tied | better | graph_prior_only | 40 | 461d446cb6b9 | 9212a0c0fc86 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | tied | better | graph_prior_only | 40 | 09ab4917bf8d | 81172bb0c9af |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | tied | better | graph_prior_only | 40 | 90f464893b7e | 8c7144920d3f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | tied | better | graph_prior_only | 40 | 1590ca159240 | 016059ac6ccb |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | tied | better | graph_prior_only | 40 | 9513ba2d0320 | 6ee99e673fca |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | tied | better | graph_prior_only | 60 | ff86b5bf430e | 6c52c902737e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | tied | better | graph_prior_only | 40 | d0eaed6a97d7 | 60056d54570c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | tied | better | graph_prior_only | 40 | dc60ef1218f3 | 7f0de8c95d0c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | tied | better | graph_prior_only | 40 | 7e21bfd906d8 | 7012b02f8b07 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | tied | better | graph_prior_only | 40 | ff3018497fa8 | 4fd10ef784ea |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | tied | better | graph_prior_only | 40 | d1ab17a57447 | 9bbb93522c94 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | tied | better | graph_prior_only | 40 | e9c8dababe35 | 35060e81c42f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | tied | better | graph_prior_only | 40 | f2db254af434 | 869b10ea034f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | tied | better | graph_prior_only | 40 | 6e0b38b55676 | 3536fe55750f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | tied | better | graph_prior_only | 40 | aa939eb9a5e3 | 4130d0ac66b8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | tied | better | graph_prior_only | 40 | 515f5fcd9502 | f5595d0b3051 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | tied | better | graph_prior_only | 40 | 83daf85e7b37 | 4bc3cfd117ba |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | tied | better | graph_prior_only | 40 | 2480474b9d12 | 6d18522517a1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | tied | better | graph_prior_only | 80 | 91f3accd1be2 | 895e11ddd032 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | tied | better | graph_prior_only | 80 | f0a73ca9bce8 | bd5089a50dcd |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | tied | better | graph_prior_only | 40 | d79a12adeb0c | 478ec09c08d1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | tied | better | graph_prior_only | 80 | 5f9e1c96e564 | 416203992179 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | tied | better | graph_prior_only | 40 | aa76fd398790 | 73bafb4967b7 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | tied | better | graph_prior_only | 40 | bc74de607032 | b232e06bc0d3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | tied | better | graph_prior_only | 40 | 700b3db9b920 | a9b33b68e42d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | tied | better | graph_prior_only | 40 | 6ed2e5233197 | 2ad6f377340f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | tied | better | graph_prior_only | 40 | 84b8a4a5874b | fd40671d2442 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | tied | better | graph_prior_only | 40 | 3708abaf102d | f46274aed92c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | tied | better | graph_prior_only | 40 | 892615b6fa51 | 915f5f370f04 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | tied | better | graph_prior_only | 40 | 99504fc65072 | 61358dc4483a |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | tied | better | graph_prior_only | 40 | bbcbae8c6f91 | 6055173727a1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | tied | better | graph_prior_only | 40 | 84152fdd8199 | 65447519290f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | tied | better | graph_prior_only | 40 | 6504b159c004 | e94d140118fb |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | worse | better | graph_prior_only | 60 | 39b6c16151a9 | 084379a716d3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | tied | better | graph_prior_only | 40 | a4f0724e8b67 | cd667c06a59c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | worse | better | graph_prior_only | 60 | bfe972f33468 | bd69b5703afe |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | tied | better | graph_prior_only | 40 | 4975c2714b6b | 44282a4fa538 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | tied | better | graph_prior_only | 40 | 58b25b388ea1 | 4013a908375d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | worse | better | graph_prior_only | 70 | 00e59b4eafd7 | 76913fcf75ba |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | tied | better | graph_prior_only | 40 | 186f5ae2e490 | 878b1bbd7df7 |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | tied | better | graph_prior_only | 40 | e464bd92281a | d4918d1f0a0b |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | tied | better | graph_prior_only | 40 | 6f15a015c5b3 | f151e6093f2e |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | tied | better | graph_prior_only | 40 | 8ddac117099f | 45a4a64eee58 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | tied | better | graph_prior_only | 60 | d5a16b0dba9b | 85f795e6bc32 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | tied | better | graph_prior_only | 40 | dc5ee543ad0c | a4e5af783639 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | tied | better | graph_prior_only | 40 | a930c1ba82e1 | 0451f603a16f |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | tied | better | vector_only | 60 | b4695d2b8691 | 196ff218ce30 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | tied | better | graph_prior_only | 40 | 648340ca08d6 | 06803cbc22b6 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | worse | better | graph_prior_only | 70 | 49d0a4a2cd6a | 5600c7df64ad |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | tied | better | graph_prior_only | 40 | a960ea8e7764 | b22f70347a9c |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | tied | better | graph_prior_only | 40 | 02d59625d327 | 1200f9126c4a |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | tied | better | graph_prior_only | 40 | db2010fd538f | fec68a12c19b |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | tied | better | graph_prior_only | 40 | cf6ffbe420a5 | de801dbbface |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | tied | better | graph_prior_only | 40 | b2da35ab5708 | 7987654cefb5 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | tied | better | graph_prior_only | 40 | 37a571e292b1 | b99aa529fc53 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | tied | better | graph_prior_only | 40 | 7d368f0b1fcd | 2028d9dd77a6 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | tied | better | graph_prior_only | 40 | a4e7d6fb9e87 | 7cab96350835 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | tied | better | graph_prior_only | 40 | ba41f29b0b5c | 0d7b43436bfa |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | tied | better | graph_prior_only | 40 | 15de8c755791 | 75f8e2abc29a |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | tied | better | graph_prior_only | 40 | f0b0e3d032a4 | 7ae7899c2305 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | worse | better | graph_prior_only | 70 | c98f65f17a7e | 71314645b75b |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | tied | better | graph_prior_only | 40 | d9ae809f1e5b | 0a85db54bc14 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | tied | better | graph_prior_only | 40 | 802b28e7d016 | a702b03e7e33 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | tied | better | graph_prior_only | 40 | 43425b4c05ac | 4d49b5ef6ce7 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | tied | better | graph_prior_only | 40 | 10f5caa98f2a | 24e6245a452c |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | tied | better | graph_prior_only | 40 | 85d05f147efd | 138d5dc4b188 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | tied | better | graph_prior_only | 40 | 8b3cdc7742d0 | f0b983f1af3e |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | tied | better | graph_prior_only | 40 | f7db21fa92b1 | fd918db0dd53 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | tied | better | graph_prior_only | 40 | 7a57c0377894 | 1e6a71d27105 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | tied | better | graph_prior_only | 40 | d748ec853559 | 4c73e5b53752 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | tied | better | graph_prior_only | 40 | 8892927f7932 | 42bbb3598224 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | tied | better | graph_prior_only | 40 | b8db174ba85f | e0336a8ef091 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | tied | better | graph_prior_only | 40 | 84b939de712b | de18ade92898 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | tied | better | graph_prior_only | 40 | 490408027b3e | 729a71114256 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | tied | better | graph_prior_only | 40 | f86a5accccd0 | 8aa3d67635d0 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | tied | better | graph_prior_only | 40 | 0d1ad5a3af27 | 1ecbd13a53e1 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | tied | better | graph_prior_only | 40 | bece40e2da69 | 6688f7efbe75 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | tied | better | graph_prior_only | 40 | 5ecc8531b663 | be7f46b5ae2c |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | tied | better | graph_prior_only | 40 | 775f9ce8d5b9 | f3da097eae88 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | tied | better | graph_prior_only | 40 | 778dac00e7ec | debdea735b17 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | tied | better | graph_prior_only | 40 | e48cd7cf08d8 | e73dc2fb034c |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | tied | better | graph_prior_only | 40 | 28d3084baeca | 87a562c3ab08 |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | tied | better | graph_prior_only | 40 | d0f385fc983c | ef25e547c15e |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | tied | better | graph_prior_only | 40 | b380543736dc | f9a37b20ac1e |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-de6291c835fe642572f46aad5198980aaf14c342250bfe9aa69d10fa85666f7a |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-7ef0f3449b3ab28ace0bf6aaad5178ef07addeae1ae1478530845ff438728512 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-4b4b2469b27ef04549f7f481fd3609938400764c0080c06efceffbbf328f6df6 |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-d597597037538681b7032730e09f2cb25ba496797420d7571cce38463a134202 |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ff9a5d7fa04a6ea40a5b485e20bfab28357447ca80fcda5e750a35c8fdaa5d83 |
| worked-traces | worked-traces.md | none | sha256-24d4eb5b9896628de4cc7cbe80a8a3cabb05be945ee045e7af5b9b5e86f893e2 |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-74cc09754bf8661e8011395a81b2e7ecd9466cb00a24233145dc9f76b0dcd43b |
