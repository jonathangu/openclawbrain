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
- learned_route tie-or-better vs graph_prior_only (traces): 372/403 (0.923077)
- learned_route vs graph_prior_only (traces): 1 better, 371 tied, 31 worse
- learned_route tie-or-better vs graph_prior_only (turns): 372/403 (0.923077)
- learned_route vs graph_prior_only (turns): 1 better, 371 tied, 31 worse
- regressions vs graph_prior_only: 31/403 (0.076923)
- regressions vs no_brain floor: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 18/832 required-context phrases vs graph_prior_only 54/832
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- activation precision: explicit learned-route activation precision is 1/403 across 403 observed candidate turns
- activation precision proxy: selection-divergence proxy activation precision is 1/403 against graph_prior_only
- success-adjusted economics: learned_route used 169 estimated prompt tokens, 0.000211 estimated prompt USD, and 10 ms serve-path latency per incremental win vs graph_prior_only 293, 0.000366, and 10
- fail-open: observed 0/403 degraded learned_route turns in this replay lane

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 7 | 403 |
| graph_prior_only | 395 | 395 |
| learned_route | 1 | 365 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | worse | better | graph_prior_only | 60 | 6ab6c6580b36 | 15cf5d5481a7 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | tied | better | graph_prior_only | 40 | 1921fd549932 | 049df50f42f2 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | worse | better | graph_prior_only | 70 | b752be9d0ae3 | e646d7e8876b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | worse | better | graph_prior_only | 80 | d12ad06e9a0c | 2a0b18c59cc0 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | tied | better | graph_prior_only | 40 | 04594d515b61 | 0b4a5dc91604 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | tied | better | graph_prior_only | 60 | cc68456247e3 | 0873679a0a4b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | worse | better | graph_prior_only | 60 | 33305710cf03 | b111b46583d8 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | worse | better | graph_prior_only | 60 | 82a86f743b20 | 8b92a08268a5 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | worse | better | graph_prior_only | 60 | 4316b7392d62 | 4d64791b8f1d |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | tied | better | graph_prior_only | 40 | 196f3e03d4fe | ca4505dcec8b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | tied | better | graph_prior_only | 60 | a6d3fda2626f | 0fe456715bc8 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | tied | better | graph_prior_only | 40 | fea5ad99d517 | 66588ae52953 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | tied | better | graph_prior_only | 40 | 9720b9e6ce4c | 8c1467aabc88 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | tied | better | graph_prior_only | 40 | 45e9ce2ada12 | a4b2dc6967f3 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | tied | better | graph_prior_only | 40 | 16d015ecf652 | 0d30e035ddf0 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | tied | better | graph_prior_only | 40 | 5c5223a64196 | 0c07fcd66a8a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | worse | better | graph_prior_only | 60 | 627b10ff490a | 74eb85d195b3 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | tied | better | graph_prior_only | 40 | 1f779fd11fc3 | 42ba262a2e0a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | tied | better | graph_prior_only | 40 | 28ed87c780c1 | 8af3243b1eb2 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | tied | better | graph_prior_only | 40 | 9c958a404e4e | 317b299860c5 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | tied | better | graph_prior_only | 40 | 401b92d30fc4 | e60a3353ee5d |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | worse | better | graph_prior_only | 100 | 2cda03ea5e7d | 0df531b0d534 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | tied | better | graph_prior_only | 40 | 2d326462b1e4 | ab2d75d7defc |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | tied | better | graph_prior_only | 40 | 48c30e71cdf5 | 0450a8854f4e |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | worse | better | graph_prior_only | 100 | c17efabdbf0e | 8007391e80d8 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | tied | better | graph_prior_only | 40 | b7716b447c6d | 6c01870ae34a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | tied | better | graph_prior_only | 40 | 72ed0ffe3a09 | e5c884c897d7 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | worse | better | graph_prior_only | 100 | 6bec60098e8c | cffc5bfe3f43 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | tied | better | graph_prior_only | 40 | 4d8b705d4be5 | a90d2f2cdfa1 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | tied | better | graph_prior_only | 40 | 0d6b0a70b098 | 098bea7d6c7f |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | tied | better | graph_prior_only | 40 | b6cad2ff4eaa | f0fd83604a76 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | tied | better | graph_prior_only | 40 | 91092ea90dc1 | e01712082c12 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | worse | better | graph_prior_only | 60 | fa2412b3e38c | 41c7b768e9b9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 3ef6995816c9 | 90357b6a871a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | tied | better | graph_prior_only | 40 | 9031f9e577d4 | f80d7020b4ab |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | tied | better | graph_prior_only | 40 | a0bdf2cb5a53 | 909d173ae2ae |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | tied | better | graph_prior_only | 40 | be955f04c8d1 | 008d816e264e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | tied | better | graph_prior_only | 40 | 616a1b114640 | 2a650637428e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | tied | better | graph_prior_only | 40 | 09dea11d9673 | d1c6cc4b0a98 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | tied | better | graph_prior_only | 40 | 1b92c190b005 | ba39769bf96b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | tied | better | graph_prior_only | 40 | 9a5995e474f2 | e2e07f06af21 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | tied | better | graph_prior_only | 40 | d6172699753e | d19916ef8346 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | tied | better | graph_prior_only | 40 | d36246fcf158 | 24fdcf30b7d5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | tied | better | graph_prior_only | 40 | b780eac67532 | e6168c244180 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | tied | better | graph_prior_only | 40 | 4b8b8e93e40b | a55b3a78233e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | tied | better | graph_prior_only | 40 | 3726c2d430e9 | b1c2a150c2bc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | tied | better | graph_prior_only | 40 | 9ba3fe29e0e4 | 1cc92a2ab08d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | tied | better | graph_prior_only | 40 | 08e93f47a04e | c8be769772ca |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | tied | better | graph_prior_only | 40 | 6620936aa966 | 4a369615bec6 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | tied | better | graph_prior_only | 40 | d80a87bb1fbe | ed8bb07b3a7b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | tied | better | graph_prior_only | 60 | ea1d2bfceaf6 | 210f15ab0f6a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | tied | better | graph_prior_only | 40 | 3f15b8a5f985 | 88a61f07c248 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | tied | better | graph_prior_only | 40 | e8a7389967ca | d8c8701a2aec |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | worse | better | graph_prior_only | 60 | 631122df937e | 267bb6d2b2d7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | tied | better | graph_prior_only | 40 | 24a91e987e1f | 048a503cc824 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | tied | better | graph_prior_only | 40 | 31cdb8e20794 | 0d4653d49f26 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | tied | better | graph_prior_only | 40 | 4b15c57fc91b | f67777629841 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | tied | better | graph_prior_only | 40 | 599bfcffdb44 | 6f358d8cb7f4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | tied | better | graph_prior_only | 40 | 31fe4563a1e4 | 63428f7cb7ec |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | tied | better | graph_prior_only | 40 | b5f61c5a863a | ebecd77f0096 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | tied | better | graph_prior_only | 40 | 8dfe4bdab5c1 | b4352106fa73 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | tied | better | graph_prior_only | 40 | 32586eb8c438 | ef2db2cd9a87 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | tied | better | graph_prior_only | 40 | aea6dc4d215f | 7bb7131d299e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | tied | better | graph_prior_only | 40 | 76900bf743f8 | 04d85c3158e9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | tied | better | graph_prior_only | 40 | 4b9d42f04abe | 13d6836106a9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | tied | better | graph_prior_only | 40 | 1139e139e4b5 | 4c34cdb641ef |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | tied | better | graph_prior_only | 40 | 1bc60240949b | b61fe118a1d2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | tied | better | graph_prior_only | 40 | 2a78e9ffcb8e | 587d1ecb7b8d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | tied | better | graph_prior_only | 60 | 5404a2bf403e | a8eaafe0429e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | tied | better | graph_prior_only | 40 | 7e99de97dd14 | 8eaa7fbcb4de |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | tied | better | graph_prior_only | 40 | d9c4261b7f01 | 5bf9b0d98d41 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | tied | better | graph_prior_only | 40 | 91ba45022124 | f9497ae0ea47 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | tied | better | graph_prior_only | 40 | 48cf7e8f2b19 | 3761f78b6534 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | tied | better | graph_prior_only | 40 | 7a67aef9dc56 | 03a1f1dceaad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | tied | better | graph_prior_only | 40 | d9174ef739d2 | 5e1fab06eca3 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | tied | better | graph_prior_only | 40 | 911c5d0e45e4 | 38bdb46975d0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | tied | better | graph_prior_only | 40 | cb18e1256914 | 9f44831f263e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | worse | better | graph_prior_only | 60 | 2e8e9984af13 | 3faccddaee2a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | tied | better | graph_prior_only | 40 | ca4982cd0984 | d3286363e67a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | tied | better | graph_prior_only | 40 | efd5998b7c8e | 5ba0b529152a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | tied | better | graph_prior_only | 40 | ef4ed4451fd8 | 8fd749aa6701 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | tied | better | graph_prior_only | 40 | 669936e90220 | df2b68c28abe |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | tied | better | graph_prior_only | 40 | 445a5e5effdb | c1e6137eb5c4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | tied | better | graph_prior_only | 40 | de47163cad6d | ce8d594ee7e9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | tied | better | graph_prior_only | 40 | aa11fb24e124 | 999868909f3e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | tied | better | graph_prior_only | 40 | c952338f4d68 | 75669052ea9a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | tied | better | graph_prior_only | 40 | 1f665f87cc95 | 9fbe0424afc5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | tied | better | graph_prior_only | 40 | aee2f5b84373 | 7b7b93d5cb65 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | tied | better | graph_prior_only | 40 | 8684c12ec2b1 | 0817c00785eb |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | tied | better | graph_prior_only | 40 | 4e1d5b62276c | cd23a2a8b9c1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | tied | better | graph_prior_only | 40 | c460a6edc737 | d8da044eb3f8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | tied | better | graph_prior_only | 40 | 080ed9f9b801 | 51c7e05d1ed0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | tied | better | graph_prior_only | 40 | 1986aba0d4f0 | b218ade7730d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | tied | better | graph_prior_only | 40 | dd9be77f8fa2 | b16de6f274d8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | tied | better | graph_prior_only | 70 | 31c1280cc57a | 1a9287223eab |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | tied | better | graph_prior_only | 40 | 821da3984021 | 20c83a387e56 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | tied | better | graph_prior_only | 40 | 71542f41bbf8 | 6e0e0b148341 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | tied | better | graph_prior_only | 40 | 62a9dd9c9a92 | aa9a6effb54d |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | tied | better | graph_prior_only | 40 | 384109e1b0dc | 1852c40a7145 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | tied | better | graph_prior_only | 40 | 0a9991647a8f | e3a457959d95 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | tied | better | graph_prior_only | 40 | a663f0f893bf | d832c6023551 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | tied | better | graph_prior_only | 40 | 6f90b113e248 | 9e8f7f237f96 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | tied | better | graph_prior_only | 40 | c9c2e16f8283 | b88cd7a98fd1 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | tied | better | graph_prior_only | 40 | b6fa430e5032 | 7d194e429739 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | tied | better | graph_prior_only | 40 | 4a97c9db0a3c | 1d13581d20f1 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | tied | better | graph_prior_only | 40 | 8e4e1366395c | 4ff1f9679bfa |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | tied | better | graph_prior_only | 40 | e56a7b20161d | 479c9b6bd04f |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | tied | better | graph_prior_only | 40 | 78a32bc5ef20 | a0621fa057a3 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | tied | better | graph_prior_only | 40 | 262ac10294dc | 9a3f17819272 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | tied | better | graph_prior_only | 40 | 69d5b098423b | 0c1d2070c1d6 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | worse | better | graph_prior_only | 60 | 097a3208f6fd | c7830649c10d |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | tied | better | graph_prior_only | 40 | 83c5e7f09c48 | 4e5ec02e6b68 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | tied | better | graph_prior_only | 40 | 45bc534fc3a7 | ec6e8b33b766 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | tied | better | graph_prior_only | 40 | c0caccd8fc4e | 022caf7dbe13 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | worse | better | graph_prior_only | 100 | d193a8058d92 | 1e0b9277137c |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | tied | better | graph_prior_only | 40 | 5e800be12a6f | 5677bb90cac8 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | tied | better | graph_prior_only | 40 | ce14751bc0cc | ba6a8213193c |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | tied | better | graph_prior_only | 40 | ec64cf2476c4 | a622c22298ec |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | tied | better | graph_prior_only | 70 | 10cf961d0860 | 4b9846f08891 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | tied | better | graph_prior_only | 40 | b312a8afd3aa | 8268e8f590f9 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | tied | better | graph_prior_only | 40 | fa1c8041c4f3 | daa33196cf1a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | tied | better | graph_prior_only | 40 | d6e52ec04112 | a3c4cc93084a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | tied | better | graph_prior_only | 40 | 672b7b859918 | 4d5550987923 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | tied | better | graph_prior_only | 40 | 4e7a47149db2 | d2742938c213 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | tied | better | graph_prior_only | 40 | 43af5f1248d0 | 00a235e8a58f |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | tied | better | graph_prior_only | 40 | 24d3e41b2e37 | e206ba924576 |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | worse | better | graph_prior_only | 100 | e30283cafd8a | 8b079c20a76c |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | tied | better | graph_prior_only | 40 | b8e7ee7e9d22 | 33513c341b71 |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | tied | better | graph_prior_only | 40 | 75feeee14b86 | db1459cae874 |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | tied | better | graph_prior_only | 40 | 4516b6537316 | 7ad0bf240197 |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | worse | better | graph_prior_only | 80 | dda0bc78deb5 | 957eaf3fbab6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | d68027de8d10 | 74514ba9b1db |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | tied | better | graph_prior_only | 40 | 99ef2514b525 | a786dc09f88c |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | tied | better | vector_only | 60 | bab9812d0702 | 8f48359130a5 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | tied | better | graph_prior_only | 40 | a45a9df73f12 | a4449827185a |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | tied | better | graph_prior_only | 40 | 1761ea5356bb | 66b90acc9da9 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | tied | better | graph_prior_only | 40 | 4c2c37f113f3 | 0b0cac34c5cc |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | tied | better | graph_prior_only | 40 | 6a15c9d5c9f6 | 299442dafe91 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | tied | better | graph_prior_only | 40 | 107dadad5291 | 83a3132634fc |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | tied | better | graph_prior_only | 40 | 05f0fd2bafc5 | 19cd7428d29e |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | tied | better | graph_prior_only | 40 | 9faeb2ff7cf1 | cb59ef6653ac |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | tied | better | graph_prior_only | 40 | 8369c8398e1e | 95e91557ef5c |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | tied | better | graph_prior_only | 40 | 9f7cc5313a17 | d6bebbabc936 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | tied | better | graph_prior_only | 40 | 16d27f02cb94 | 67bc5b08626f |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | tied | better | graph_prior_only | 40 | db173aef1ac4 | 568646b7968c |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | tied | better | graph_prior_only | 40 | e73f69453056 | 02ed5b603292 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | tied | better | graph_prior_only | 40 | 8a568e809025 | 309bffe342ad |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | tied | better | graph_prior_only | 40 | 6e0f42a847af | 87612f1370a6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | tied | better | graph_prior_only | 40 | b55d5603219c | cd64d18349bf |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 1cb848e72747 | a8de3e1e3dc7 |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | tied | better | vector_only | 80 | 09d597cdd081 | 61eec6489a6a |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | worse | better | graph_prior_only | 60 | 294b61803bd7 | 95cd87b52707 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | tied | better | graph_prior_only | 40 | 5b5dd020ed33 | c87f8a42d28c |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | tied | better | graph_prior_only | 40 | a0f8ea7809c5 | 0cc06d8470ee |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | tied | better | graph_prior_only | 40 | 7e03deabf8cc | 71f828e68841 |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | tied | better | graph_prior_only | 40 | b6e12a490d88 | 54ff3f690f04 |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | tied | better | graph_prior_only | 40 | 86a285683bfd | f163e0c6420f |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | tied | better | graph_prior_only | 40 | 8f0ebdeb52d9 | e6d06d4edad4 |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | worse | better | graph_prior_only | 60 | d4e6645065f8 | 235ac30766fc |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | tied | better | graph_prior_only | 40 | 92f5ed2edd4a | 8eec058e861b |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | worse | better | graph_prior_only | 100 | c82e9a237a41 | aef5b04a7893 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | tied | better | vector_only | 70 | 1747f336da20 | 5990c8cbb76b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | tied | better | graph_prior_only | 40 | be7ba3d9c101 | c6a286e01a84 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | tied | better | graph_prior_only | 40 | 8590df941ebb | 58f4ab05f4e1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | tied | better | graph_prior_only | 40 | 0d9c633ede48 | b6bd4b22e3e4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | tied | better | graph_prior_only | 40 | f2a61e9538c1 | 7beda594be71 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | tied | better | graph_prior_only | 40 | be62eef6bd63 | d5b017cdb07a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | tied | better | graph_prior_only | 40 | b42fe61cabeb | 49013d857e5e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | tied | better | graph_prior_only | 40 | e0a2399881d7 | e224221154a0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | tied | better | graph_prior_only | 40 | c117a8ac2eea | 98c476060fe7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | tied | better | graph_prior_only | 40 | 383707419932 | e0bc75317510 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | tied | better | graph_prior_only | 40 | c80e1b2d0a0b | 780681d3bfaa |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | tied | better | graph_prior_only | 40 | d949bd35ae00 | 6a14bdc1aff8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | tied | better | vector_only | 100 | 8d68a0ee603f | f5b50af1cea3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | worse | better | graph_prior_only | 100 | 4869acd71550 | db972f17752f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | tied | better | graph_prior_only | 40 | 74389ef44925 | e23928b4bd35 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | tied | better | graph_prior_only | 40 | 2c81f32c8593 | 1a292842b897 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | tied | better | graph_prior_only | 40 | 7334480546a3 | 54ebe275a4d8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | tied | better | graph_prior_only | 40 | b002a9df5d07 | 28f26cae824b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | tied | better | graph_prior_only | 40 | 0c7122bf20c9 | fcf3da4c6791 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | tied | better | graph_prior_only | 40 | b0b651136b80 | 36f5cebb711b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | tied | better | graph_prior_only | 40 | fc8fff6c27c3 | c5c05808a2f5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | tied | better | graph_prior_only | 40 | 1b6d1144769c | adb8302938a8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | tied | better | graph_prior_only | 40 | 717df485683a | ae18f3a83d0e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | tied | better | graph_prior_only | 40 | bdb2b6e91fb3 | 00a55d4c7c57 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | tied | better | graph_prior_only | 40 | d81dee9d993b | ca26af77da05 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | tied | better | graph_prior_only | 40 | acc3f0e071e7 | a0d94340725a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | tied | better | graph_prior_only | 40 | 7e005db8d58f | 5e850318f431 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | tied | better | graph_prior_only | 40 | 910f1acf4978 | b10ef6fc4e28 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | tied | better | graph_prior_only | 40 | ee8b7a6beeaa | fed49c26a94e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | worse | better | graph_prior_only | 100 | aaaaa93a21aa | 2d207858aa1d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | tied | better | graph_prior_only | 40 | 774a3109ac9e | 3316c05f3f97 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | tied | better | graph_prior_only | 40 | 2402d5bff8e9 | 549dfc413095 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | tied | better | graph_prior_only | 40 | ffcf576c5dbb | be7400c2bfb1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | tied | better | graph_prior_only | 40 | 1f464faabd4e | 1724ebafaa98 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | tied | better | graph_prior_only | 40 | 9592bea1f702 | 2a2c9e0cac70 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | tied | better | graph_prior_only | 40 | 61cddb9af54c | cc9576319d4c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | tied | better | graph_prior_only | 40 | c3ff4c0c35bc | eceebc35546c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | tied | better | graph_prior_only | 40 | 6005c7e0f8ed | af4409b33951 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | tied | better | graph_prior_only | 40 | 6a0335e4020b | bccaaf5edc34 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | tied | better | graph_prior_only | 40 | 674f898c8a31 | 8027409513e4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | tied | better | graph_prior_only | 40 | 559b0e4c4eb9 | d1574a51aaca |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | tied | better | vector_only | 100 | 071c7400af1c | fb1b34a1bfa0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | tied | better | graph_prior_only | 40 | 3c43df8b4fd7 | a446cbccea9c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | tied | better | graph_prior_only | 40 | fb2088bfd5a9 | dae7c5f8f2b5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | tied | better | graph_prior_only | 40 | e28ee8ade446 | f7347a1b3a96 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | tied | better | graph_prior_only | 40 | 33e624bfe4a2 | 6236c86d85e8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | tied | better | graph_prior_only | 40 | fd6e443db3b8 | 52fa3e111268 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | tied | better | graph_prior_only | 40 | 1c9d0d7164bf | 2ef20041e2b2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | tied | better | graph_prior_only | 40 | eb7eac135cc3 | edf5a5525fff |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | tied | better | graph_prior_only | 40 | d56f919f62cb | 19bec722be88 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | tied | better | graph_prior_only | 40 | 4f7e643d5ec1 | 8ca93240f7b6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | tied | better | graph_prior_only | 40 | bb9cbf9fd8d7 | 1cf8232f599c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | tied | better | graph_prior_only | 40 | f31f72cb96c4 | 5ed8be6a5694 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | tied | better | graph_prior_only | 40 | 08fb4280abc9 | 22eb418e00a0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | tied | better | graph_prior_only | 40 | 81767e78dc8d | 500e5be4d608 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | tied | better | graph_prior_only | 40 | 5d861d6527ac | 08f6fa2b433d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | tied | better | graph_prior_only | 40 | 3e69cd607536 | f81589c9d87a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | tied | better | graph_prior_only | 40 | 4eeb7c24d7c2 | 949eabef596d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | tied | better | graph_prior_only | 40 | 4a8179f5cadd | dc7eb4efe922 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | tied | better | graph_prior_only | 40 | d3a96104c133 | 155094ae52e5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | tied | better | graph_prior_only | 40 | 9efc1c6fe446 | af8d31b06170 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | tied | better | graph_prior_only | 40 | dc4dee81ccd8 | b5f1419b3bbf |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | tied | better | graph_prior_only | 40 | 6e4aaeacafd9 | 3ef73429b402 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | tied | better | graph_prior_only | 40 | b3e2ff219ef5 | 048f04347f72 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | tied | better | graph_prior_only | 40 | 21fc9bbc6fc2 | 6e5eb70280cb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | tied | better | graph_prior_only | 40 | 175b5970503d | 44488f95a832 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | tied | better | graph_prior_only | 40 | 353261f72332 | 7717fcc322d3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | tied | better | graph_prior_only | 40 | 1ca0117101ff | d7af1d8f48a5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | tied | better | graph_prior_only | 40 | 586704a4d10d | 29b2c18ad1ac |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | tied | better | graph_prior_only | 40 | ca412b52ce2b | a561bd68aa16 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | tied | better | graph_prior_only | 40 | 434d5421a067 | 8ac6465fdb26 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | tied | better | graph_prior_only | 40 | 0968d587f685 | e515f7486791 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | tied | better | graph_prior_only | 40 | cfddec06abc0 | 16dfee582330 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | tied | better | graph_prior_only | 40 | 0d33827f8132 | 5d398cb43e7f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | tied | better | graph_prior_only | 40 | fd5dc397b76e | 5ebcbf9e8900 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | tied | better | graph_prior_only | 40 | d7f0250028a1 | 28412011345f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | tied | better | graph_prior_only | 40 | 3f390edb04cd | 764f94f6f065 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | tied | better | graph_prior_only | 40 | 2649a6b1312c | b2dd7d9b119d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | tied | better | graph_prior_only | 40 | 6af9d482125b | 1ec1d31c82e4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | tied | better | graph_prior_only | 40 | 80319afc3e04 | 4c779bf3683b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | tied | better | graph_prior_only | 40 | 943bc6202922 | 9e437fd3d68d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | tied | better | graph_prior_only | 40 | 59b3023b5223 | 88ad2465fc4f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | tied | better | graph_prior_only | 40 | e94ce98af629 | 6f16dff1b1ad |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | tied | better | graph_prior_only | 40 | 954267bd415d | ddab66e349e7 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | tied | better | graph_prior_only | 40 | 2329ef614da3 | 9edaa3b10a9f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | tied | better | graph_prior_only | 40 | f9b8169e3486 | ebf44f1e0edf |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | worse | better | graph_prior_only | 60 | dea188aebba9 | 1aa3f3c9b04c |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | tied | better | graph_prior_only | 60 | eef430b4f678 | fe7f8b01f8fd |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | tied | better | graph_prior_only | 40 | 94ceb10b3623 | 38bfb0530318 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | tied | better | graph_prior_only | 40 | 691724bb9ab6 | 015ed4563792 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | tied | better | graph_prior_only | 40 | 94464249fd3f | b6f4df99c056 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | tied | better | graph_prior_only | 40 | 3727d6c87e0f | 2943f5ccbed5 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | tied | better | graph_prior_only | 40 | c1802fa18c9a | 47b706e7c4aa |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | tied | better | graph_prior_only | 40 | 2e97f949852d | 03183b4228e6 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | tied | better | graph_prior_only | 40 | b11ba250ada9 | acc82388d447 |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | tied | better | graph_prior_only | 40 | 5b6ab8fd26be | d1f7bd88ac34 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | worse | better | graph_prior_only | 60 | 10ed20a70f91 | ca723012b952 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | tied | better | graph_prior_only | 40 | 7f2c2424adae | 853ace4dcd28 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | tied | better | graph_prior_only | 40 | 533d98998cea | a6760ce45bb7 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | worse | better | graph_prior_only | 60 | f8b34c09ccb6 | 1902d9f28972 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | tied | better | graph_prior_only | 40 | cd363909857d | a55f0981e1bb |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | tied | better | graph_prior_only | 40 | 679886b6acff | c374c06f0075 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | tied | better | graph_prior_only | 40 | 3f1a4bb81271 | 02c56ab488ee |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | tied | better | graph_prior_only | 40 | 2fb73afaf75d | 7e979884cb27 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | tied | better | graph_prior_only | 40 | 58ccf26f41ac | d408839de965 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | tied | better | graph_prior_only | 40 | 67fda41f5543 | e89af162feb8 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | tied | better | graph_prior_only | 40 | f690431a0462 | a72e56685dfa |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | tied | better | graph_prior_only | 40 | 896e88ae86cf | 76444a4dd480 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | tied | better | graph_prior_only | 40 | 90ba20d68ee2 | ea311c8a8448 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | tied | better | graph_prior_only | 40 | c49f958c1904 | dc06b2c2ac98 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | tied | better | graph_prior_only | 40 | 258d0daa3b25 | 77a1c1c3648b |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | tied | better | graph_prior_only | 40 | 2019c74576b8 | 968295c03bb3 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | tied | better | graph_prior_only | 40 | 159003103b17 | 0c5ff1b5ebcf |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | tied | better | graph_prior_only | 40 | b013bdb12a2c | a57f49f4d539 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | tied | better | graph_prior_only | 40 | 4c9b12d957f7 | 13e8ffba2be5 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | tied | better | graph_prior_only | 40 | 7988acbd6a1c | 9a3dafcd4983 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | tied | better | graph_prior_only | 40 | c1f85f45f0cb | d474ef36f060 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | tied | better | graph_prior_only | 40 | 3516824ab3e4 | cf8b761fa83b |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | tied | better | graph_prior_only | 40 | 350c5d1a212c | e444e5e4bf84 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | tied | better | graph_prior_only | 40 | 06b3ffac26c5 | 455d599ccb13 |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | better | better | learned_route | 60 | b252e697b03e | ab88f1a6f6fd |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | tied | better | graph_prior_only | 40 | 4575747324d0 | 8b714daed723 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | tied | better | graph_prior_only | 40 | c73b7d694c0a | d0504ee91bf4 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | tied | better | vector_only | 70 | 9b05b36e5de4 | 0ca16841ae77 |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | tied | better | graph_prior_only | 40 | 8345176219c1 | b0ba77eb6174 |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | worse | better | graph_prior_only | 60 | 744a81896989 | 17c2044d3a87 |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | tied | better | graph_prior_only | 40 | ad71f1044ad0 | c12d2ead7625 |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | tied | better | graph_prior_only | 40 | 03fbc9e61f9b | 57ad1d0a8969 |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | tied | better | graph_prior_only | 40 | 9c3f5cbe037f | 5da40ab29069 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | tied | better | graph_prior_only | 40 | 10e80319a042 | 732a8e7512a0 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | tied | better | graph_prior_only | 40 | 13488ceb3699 | 1e5d87c4cb08 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | tied | better | graph_prior_only | 40 | c9753228350f | acc1b3920323 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | tied | better | graph_prior_only | 40 | 939183cd6213 | 49948eb5856e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | tied | better | graph_prior_only | 40 | 94fa598f816b | af466f70bb8f |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | tied | better | graph_prior_only | 40 | 897283971381 | 8011836b8d2e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | tied | better | graph_prior_only | 40 | 660bf1129fa3 | 51185587c804 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | tied | better | graph_prior_only | 60 | bdbf01c4e5c5 | f7e512d8f85f |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | tied | better | graph_prior_only | 60 | ecaca9036d02 | e6154cf5d3f7 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | tied | better | graph_prior_only | 40 | 1a833171a0b8 | c998758389c8 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | tied | better | graph_prior_only | 40 | 67e24e65ee79 | 7636a35bc6bb |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | tied | better | graph_prior_only | 40 | 8ed1ba7f229c | 30f522c93f8f |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | tied | better | graph_prior_only | 40 | 6b09bf62a5ee | 26b275b9369d |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | tied | better | graph_prior_only | 40 | 60310f4a72df | ed943b644704 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | edcf357cd916 | 3b21fa36b7f5 |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | tied | better | graph_prior_only | 40 | 88f51de45743 | 3c4b690e3394 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | tied | better | graph_prior_only | 40 | d6f78807309a | 0bcd37f3fd63 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | tied | better | graph_prior_only | 40 | a2b80be3596d | 0c7dc13f3956 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | tied | better | graph_prior_only | 40 | 85faeae2ded9 | 7aef2ef1775c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | tied | better | graph_prior_only | 40 | b83cba5e6d8b | 8a28a2a51a25 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | tied | better | graph_prior_only | 40 | 1d643263606a | b40e079f73ee |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | tied | better | graph_prior_only | 40 | 1fd6fa8cd43c | 352989a25c71 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | tied | better | graph_prior_only | 40 | 2e3391c47c2d | a1109417b8b7 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | tied | better | graph_prior_only | 40 | 65563a99b761 | 282dfe18ec6e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | tied | better | graph_prior_only | 40 | 3985df17ec16 | 6c300b48fbc0 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | tied | better | graph_prior_only | 40 | 6cafa8d5c6c1 | 7c69680d9919 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | tied | better | graph_prior_only | 40 | d84976f38852 | 933c686ed9ed |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | tied | better | graph_prior_only | 40 | 936c45fe6ea4 | 6c402098b896 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | tied | better | graph_prior_only | 40 | 87a4330e994e | 08d9e6048ad2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | tied | better | graph_prior_only | 40 | f1002b6f0ba0 | 39b14b15249f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | tied | better | graph_prior_only | 40 | 0863f371e718 | 63739ac67f4b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | tied | better | graph_prior_only | 40 | c3e58d9d6a03 | f33ec77ce540 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | tied | better | graph_prior_only | 40 | ed7cb90c6cfc | bd0729f4d8ef |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | tied | better | graph_prior_only | 40 | 9da707e2f4f0 | 8fc8816c4868 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | tied | better | graph_prior_only | 60 | 8e55c6036453 | c847613d7298 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | tied | better | graph_prior_only | 40 | 487a6d699c2e | 5deb9fdb9234 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | tied | better | graph_prior_only | 40 | 3439eba50f14 | 05fa8b1e6d3d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | tied | better | graph_prior_only | 40 | f04c166a9560 | 893ca260abf3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | tied | better | graph_prior_only | 40 | 9ca30448af79 | ae1914bdc4b9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | tied | better | graph_prior_only | 40 | 12c4be022586 | fc7ee7986417 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | tied | better | graph_prior_only | 40 | 142d22ed1048 | 025c29a5be66 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | tied | better | graph_prior_only | 40 | 503d66260bb1 | f7164628594d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | tied | better | graph_prior_only | 40 | ef9fbf161b1b | 033ad40d3992 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | tied | better | graph_prior_only | 40 | 6c3bf031f338 | 8fc2fd42cd00 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | tied | better | graph_prior_only | 40 | fce7435a118a | 2e7ffb7b550a |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | tied | better | graph_prior_only | 40 | 802fd0cea934 | 81671f47ebaf |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | tied | better | graph_prior_only | 40 | 07a1a46509c7 | 96c42b44114e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | tied | better | graph_prior_only | 80 | 7681a5cc9f01 | 05d1bba470fc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | tied | better | graph_prior_only | 80 | 52f5abe10c68 | fdc5c743b8a7 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | tied | better | graph_prior_only | 40 | 5e6686ddb8fb | 393faa3f4afb |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | tied | better | graph_prior_only | 80 | 48793425eb4d | 349a5b0ce0e4 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | tied | better | graph_prior_only | 40 | 2501499b1a2d | 61a84ccebbcc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | tied | better | graph_prior_only | 40 | ce518cbdeff1 | 5583ced33f55 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | tied | better | graph_prior_only | 40 | 624816decd45 | 7196fbb16647 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | tied | better | graph_prior_only | 40 | 47106b418c9a | 7f3f97006c95 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | tied | better | graph_prior_only | 40 | b6ed9a6b2e44 | 3c2f306bfe30 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | tied | better | graph_prior_only | 40 | 60fcf2d3577d | 52d2387bf1c3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | tied | better | graph_prior_only | 40 | 0748d034d857 | 280c6abb4a65 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | tied | better | graph_prior_only | 40 | 9ca7613e9be3 | e277687fd8c8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | tied | better | graph_prior_only | 40 | 1f13151314b0 | 18eb98829f03 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | tied | better | graph_prior_only | 40 | 093dd0993ed4 | c31eb0eb6a26 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | tied | better | graph_prior_only | 40 | b5ae0ac0073d | c22a517b1fd3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | worse | better | graph_prior_only | 60 | 96f04b6c9268 | 55a7cd4c474e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | tied | better | graph_prior_only | 40 | 5cfa82cf4115 | 3c7d9da742cc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | worse | better | graph_prior_only | 60 | aabd9fd4da52 | 51f0b0a47ec1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | tied | better | graph_prior_only | 40 | 977aed56342e | d8d2bf56b0a3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | tied | better | graph_prior_only | 40 | fbada315582c | 4bac93f278ae |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | worse | better | graph_prior_only | 70 | 9c454b8b8ba6 | 8eee90f3686d |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | tied | better | graph_prior_only | 40 | 6b1db0633ff3 | 6459907dc584 |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | tied | better | graph_prior_only | 40 | c024918707d2 | 71267ac7c8ce |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | tied | better | graph_prior_only | 40 | 98bb0015714e | 0bf4452e35b0 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | tied | better | graph_prior_only | 40 | 81ed7f691ebe | e43b85c42caf |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | tied | better | graph_prior_only | 60 | 44d3774637b4 | 2ff43d0711ae |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | tied | better | graph_prior_only | 40 | 11d39f0e17bd | a18f6bf033fb |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | tied | better | graph_prior_only | 40 | 09ee308d8f9a | 77746e2ccdb7 |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | tied | better | vector_only | 60 | 7dd4e44cdbde | 88b70117fa92 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | tied | better | graph_prior_only | 40 | d71fac483cfc | ef705d13a0ab |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | worse | better | graph_prior_only | 70 | b4cadf8e009f | 14dcca9716ae |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | tied | better | graph_prior_only | 40 | a8c24d0e34a7 | 86abc7a70511 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | tied | better | graph_prior_only | 40 | bbbf02fef767 | 4a82be1dbd18 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | tied | better | graph_prior_only | 40 | 74f2b2ccc2d5 | f29c02ba576d |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | tied | better | graph_prior_only | 40 | 3bc54440c0bd | 177234da11a0 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | tied | better | graph_prior_only | 40 | 2528615c76fa | e4123fb580c4 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | tied | better | graph_prior_only | 40 | 351d8a25aed3 | 06679f7c992b |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | tied | better | graph_prior_only | 40 | 541dd0100d05 | 1283ec930261 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | tied | better | graph_prior_only | 40 | a5f24d6a47da | 762a692eb8cd |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | tied | better | graph_prior_only | 40 | 1534f4368021 | 6b546240b3d5 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | tied | better | graph_prior_only | 40 | 22682fc4fc5a | 3c8ed76116de |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | tied | better | graph_prior_only | 40 | ac1a25533f1e | 41e26a827149 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | worse | better | graph_prior_only | 70 | ece196fb29ce | 0e74741983c6 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | tied | better | graph_prior_only | 40 | c32ad48bd360 | 5235af6c877a |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | tied | better | graph_prior_only | 40 | b9a9d5205b2f | cfa4d6742726 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | tied | better | graph_prior_only | 40 | d686a167f876 | dd59b7b17b7a |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | tied | better | graph_prior_only | 40 | 82441119b2b5 | 50b4f68b82c8 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | tied | better | graph_prior_only | 40 | bbb77b9f1926 | ef63b52ae5c2 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | tied | better | graph_prior_only | 40 | c5d5412a525e | 2011190ed857 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | tied | better | graph_prior_only | 40 | 7c119842c47b | 82edb836de3d |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | tied | better | graph_prior_only | 40 | b69e7fa9af3c | 707eccece7db |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | tied | better | graph_prior_only | 40 | 3966ad9eb2db | f54a121234cb |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | tied | better | graph_prior_only | 40 | 728a2ef61bdf | f9977b6e2b3a |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | tied | better | graph_prior_only | 40 | ec2fce79d0cc | c026a7e75ca8 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | tied | better | graph_prior_only | 40 | 79ba651e7e68 | 1e53e60bff2d |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | tied | better | graph_prior_only | 40 | e2f997e652ea | 38781006fff1 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | tied | better | graph_prior_only | 40 | d8a733bd0964 | cfa32cf73c27 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | tied | better | graph_prior_only | 40 | 2ce29cdb872a | 805bf64104f1 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | tied | better | graph_prior_only | 40 | ba968d93d0ba | f0cabcdecfd8 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | tied | better | graph_prior_only | 40 | b557afbdb731 | 155164053edc |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | tied | better | graph_prior_only | 40 | 68b47bb19871 | 58af0f6814e2 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | tied | better | graph_prior_only | 40 | d7722348d2c3 | ec26cdb820d5 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | tied | better | graph_prior_only | 40 | 467fabece52a | 5c9e2416d2c5 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | tied | better | graph_prior_only | 40 | d349c5de0f68 | 06d104370caa |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | tied | better | graph_prior_only | 40 | 8f44b1987d8e | b17c07fbd3df |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | tied | better | graph_prior_only | 40 | e66acd779620 | 03fef889ac87 |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-f484d93717679fc0155b52e3b1eb7b3778a41fd8641229bb9cab7c1af27fc669 |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-ac65b8da7c4252230308559fcab59fa1eaf61d6821ded638d123a2c20310efb4 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-f3ed28f4ed2419d887773d06ed7beda9c6b07148190dc22f77de898599806fa9 |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-dbcd4e03471959825224d2a74f2d4e1f5b09187c7ac978471e78e5007c06093d |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-1f3f90788b9e78db2e03f2681e95fe9a58bce6531b1d15db0f20358b6e76d5a1 |
| worked-traces | worked-traces.md | none | sha256-c46a140a79eb21edf2310fa2b2c22e791bb8b707af35f96264d09b450cc71aa3 |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-0a67cc863a6b2ca4c5ae273e9a2863687a8fefbc31da452b0ad102c542d26c40 |
