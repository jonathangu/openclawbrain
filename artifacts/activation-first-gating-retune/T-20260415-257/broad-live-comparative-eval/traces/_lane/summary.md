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
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | worse | better | graph_prior_only | 60 | 62ed6ad3eb02 | 15cf5d5481a7 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | tied | better | graph_prior_only | 40 | a7c7621ce9a9 | 049df50f42f2 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | worse | better | graph_prior_only | 70 | 5435aa8bd8fa | e646d7e8876b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | worse | better | graph_prior_only | 80 | a5e5ea8213f7 | 6882305878ee |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | tied | better | graph_prior_only | 40 | 90432ddf2df1 | cf3d2194c11e |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | tied | better | graph_prior_only | 60 | bc16cf88ab53 | 9d7430f8bdef |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | worse | better | graph_prior_only | 60 | d38468ce17eb | 85f291112f20 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | worse | better | graph_prior_only | 60 | 5379e338bb48 | 6139c3db2b26 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | tied | better | graph_prior_only | 60 | ee9bfaacd755 | 74c2938c8674 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | tied | better | graph_prior_only | 40 | 13fb481dfe5b | ae40af128be9 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | tied | better | graph_prior_only | 60 | 5606a9bf41f0 | d687b0b61532 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | tied | better | graph_prior_only | 40 | fedd0926cb9d | 12f826a86e1b |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | tied | better | graph_prior_only | 40 | d8ad1110ce2f | 8641d7befdd1 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | tied | better | graph_prior_only | 40 | 178532c5242e | 9a382ec1c0f9 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | tied | better | graph_prior_only | 40 | a93961aeacf3 | 09fb58f6d603 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | tied | better | graph_prior_only | 40 | 35b4bee3b72b | 38ea451ec625 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | worse | better | graph_prior_only | 60 | d54034238039 | 4965f5bd6c2c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | tied | better | graph_prior_only | 40 | 927aafb2719e | 2764709e96c3 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | tied | better | graph_prior_only | 40 | f92b51683ffd | 6ff13651ac6c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | tied | better | graph_prior_only | 40 | 1bc54079c84d | 6c95ac812466 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | tied | better | graph_prior_only | 40 | 8fa3a6be794f | 0b0272acc964 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | worse | better | graph_prior_only | 100 | dbf849fe71b8 | 2164028857c9 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | tied | better | graph_prior_only | 40 | 4ad05f97bb2c | b5df5da0b4b8 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | tied | better | graph_prior_only | 40 | 663e5c633d3d | eac75c6cffde |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | worse | better | graph_prior_only | 100 | 421e48c95a9b | de153b430b5a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | tied | better | graph_prior_only | 40 | b86b48fdfc95 | a64ee88f8b58 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | tied | better | graph_prior_only | 40 | 819ef3efb45d | f9ad179e2d20 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | worse | better | graph_prior_only | 100 | 184e347d0c89 | 719603297956 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | tied | better | graph_prior_only | 40 | 32ef774aa01e | 8026de48e70e |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | tied | better | graph_prior_only | 40 | e4aac8b9b664 | 84cfed3ecc19 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | tied | better | graph_prior_only | 40 | 4cdaa58ed82a | bfcb51fd825d |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | tied | better | graph_prior_only | 40 | a34fd9ef8db1 | 6161320c80d7 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | worse | better | graph_prior_only | 60 | bbd0a009906b | ce2fc023fc97 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | bef36ac76a33 | 90357b6a871a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | tied | better | graph_prior_only | 40 | 70dc15485d39 | f80d7020b4ab |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | tied | better | graph_prior_only | 40 | bc15cf3267e6 | 909d173ae2ae |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | tied | better | graph_prior_only | 40 | dbc49fb7cbf7 | 008d816e264e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | tied | better | graph_prior_only | 40 | 6baee68fb955 | 2a650637428e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | tied | better | graph_prior_only | 40 | 736135033732 | d1c6cc4b0a98 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | tied | better | graph_prior_only | 40 | 14158d674934 | ba39769bf96b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | tied | better | graph_prior_only | 40 | d95a265e7940 | e2e07f06af21 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | tied | better | graph_prior_only | 40 | 1446d6c19d80 | d19916ef8346 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | tied | better | graph_prior_only | 40 | 98fe50dbbaca | 24fdcf30b7d5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | tied | better | graph_prior_only | 40 | add1d6f2ffd0 | e6168c244180 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | tied | better | graph_prior_only | 40 | 3b162fba7229 | a55b3a78233e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | tied | better | graph_prior_only | 40 | efd3eac6cc43 | b1c2a150c2bc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | tied | better | graph_prior_only | 40 | a8ceb2f4e168 | 1cc92a2ab08d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | tied | better | graph_prior_only | 40 | 66d74bdaf9d5 | c8be769772ca |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | tied | better | graph_prior_only | 40 | 18a32fda87a9 | 4a369615bec6 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | tied | better | graph_prior_only | 40 | 4296de74f025 | ed8bb07b3a7b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | tied | better | graph_prior_only | 60 | 794df61e1e91 | 210f15ab0f6a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | tied | better | graph_prior_only | 40 | ccd3e6208533 | 88a61f07c248 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | tied | better | graph_prior_only | 40 | 5aa1af00f093 | d8c8701a2aec |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | worse | better | graph_prior_only | 60 | e472a7ab7e94 | 267bb6d2b2d7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | tied | better | graph_prior_only | 40 | c991f4a724fd | 048a503cc824 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | tied | better | graph_prior_only | 40 | 69c97edc01d0 | 0d4653d49f26 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | tied | better | graph_prior_only | 40 | fed7a7ee51c9 | f67777629841 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | tied | better | graph_prior_only | 40 | ba5aee84aa51 | 6f358d8cb7f4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | tied | better | graph_prior_only | 40 | 0ebea1a78f91 | 63428f7cb7ec |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | tied | better | graph_prior_only | 40 | dde28db6181e | ebecd77f0096 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | tied | better | graph_prior_only | 40 | e9073bfdad6d | b4352106fa73 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | tied | better | graph_prior_only | 40 | 77470e7c6815 | ef2db2cd9a87 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | tied | better | graph_prior_only | 40 | ffed86fb8dfd | 7bb7131d299e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | tied | better | graph_prior_only | 40 | 5e79b9ff49f7 | 04d85c3158e9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | tied | better | graph_prior_only | 40 | e0906d69ca47 | 13d6836106a9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | tied | better | graph_prior_only | 40 | 6aebd2b344ee | 4c34cdb641ef |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | tied | better | graph_prior_only | 40 | 4535a4ad1627 | b61fe118a1d2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | tied | better | graph_prior_only | 40 | c0c608b98020 | 587d1ecb7b8d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | tied | better | graph_prior_only | 60 | 081671a90675 | a8eaafe0429e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | tied | better | graph_prior_only | 40 | 7b689a2a97a1 | 8eaa7fbcb4de |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | tied | better | graph_prior_only | 40 | 2b4ad26b6d96 | 5bf9b0d98d41 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | tied | better | graph_prior_only | 40 | e0e079a2f567 | f9497ae0ea47 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | tied | better | graph_prior_only | 40 | 2a193950666c | 3761f78b6534 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | tied | better | graph_prior_only | 40 | c573dea723e6 | 03a1f1dceaad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | tied | better | graph_prior_only | 40 | e59193d42473 | d45870d79a2f |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | tied | better | graph_prior_only | 40 | 9a4180c56033 | 38bdb46975d0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | tied | better | graph_prior_only | 40 | 43fbebe41404 | 9f44831f263e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | worse | better | graph_prior_only | 60 | d85b81277adc | 3faccddaee2a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | tied | better | graph_prior_only | 40 | c9987ad35ff4 | d3286363e67a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | tied | better | graph_prior_only | 40 | 7903dad319a0 | 5ba0b529152a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | tied | better | graph_prior_only | 40 | 2e3227391739 | 8fd749aa6701 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | tied | better | graph_prior_only | 40 | c88a48273f5c | df2b68c28abe |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | tied | better | graph_prior_only | 40 | 5976ed55e31f | c1e6137eb5c4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | tied | better | graph_prior_only | 40 | e58ef8beb6af | 6412f1b8568a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | tied | better | graph_prior_only | 40 | ee52e992342e | 999868909f3e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | tied | better | graph_prior_only | 40 | 685808086be8 | 75669052ea9a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | tied | better | graph_prior_only | 40 | e07aba8aab46 | 9fbe0424afc5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | tied | better | graph_prior_only | 40 | 84e82082f054 | 7b7b93d5cb65 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | tied | better | graph_prior_only | 40 | e905c5c00a64 | 0817c00785eb |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | tied | better | graph_prior_only | 40 | e0b725bfbf3f | cd23a2a8b9c1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | tied | better | graph_prior_only | 40 | 626f9bbcbea8 | d8da044eb3f8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | tied | better | graph_prior_only | 40 | 310e240dff60 | 51c7e05d1ed0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | tied | better | graph_prior_only | 40 | 8753107cd941 | b218ade7730d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | tied | better | graph_prior_only | 40 | 8fb1440a695b | b16de6f274d8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | tied | better | graph_prior_only | 70 | f80c6a7fb6f3 | 1a9287223eab |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | tied | better | graph_prior_only | 40 | 61461b3cacca | 3f3cdd121184 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | tied | better | graph_prior_only | 40 | 7fc869e43b6c | 09b2f2fb74c3 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | tied | better | graph_prior_only | 40 | 2d599594aafd | 5d83692c90cb |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | tied | better | graph_prior_only | 40 | 2477d7924f97 | e86b520fd57b |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | tied | better | graph_prior_only | 40 | c31d10ac5d1e | ddc076b79805 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | tied | better | graph_prior_only | 40 | 13e1d9853cc7 | 23e6b6a94c9e |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | tied | better | graph_prior_only | 40 | 45bf0960236f | d769d8137347 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | tied | better | graph_prior_only | 40 | 9f0124fcd8bc | 0a23fb661f5c |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | tied | better | graph_prior_only | 40 | f91d56701d74 | dc9f9b5058b7 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | tied | better | graph_prior_only | 40 | fef2e810b1db | 5d43f311142b |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | tied | better | graph_prior_only | 40 | 5d0838e59880 | 69c9950e79c3 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | tied | better | graph_prior_only | 40 | e28e6c2fe9f6 | 17d4b972a68e |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | tied | better | graph_prior_only | 40 | f6ca142f9acc | a0621fa057a3 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | tied | better | graph_prior_only | 40 | fde8bdac3c02 | 9a3f17819272 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | tied | better | graph_prior_only | 40 | 44f9f173ffea | 0c1d2070c1d6 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | worse | better | graph_prior_only | 60 | 7d70059094ea | c7830649c10d |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | tied | better | graph_prior_only | 40 | f548cbf1fdeb | 4e5ec02e6b68 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | tied | better | graph_prior_only | 40 | 25dc4147caea | ec6e8b33b766 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | tied | better | graph_prior_only | 40 | d6ad20e067ad | 022caf7dbe13 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | worse | better | graph_prior_only | 100 | 02ca7a8da601 | 1e0b9277137c |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | tied | better | graph_prior_only | 40 | 22b19d32118b | 5677bb90cac8 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | tied | better | graph_prior_only | 40 | 63bc56f55d19 | ba6a8213193c |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | tied | better | graph_prior_only | 40 | aaed1c47b763 | a622c22298ec |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | tied | better | graph_prior_only | 70 | dca3291204df | 4b9846f08891 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | tied | better | graph_prior_only | 40 | b5bcf19ea364 | 8268e8f590f9 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | tied | better | graph_prior_only | 40 | d649363f0f97 | daa33196cf1a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | tied | better | graph_prior_only | 40 | 410375c74ca2 | a3c4cc93084a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | tied | better | graph_prior_only | 40 | bfa6e798bf0f | 4d5550987923 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | tied | better | graph_prior_only | 40 | 77715f05b51b | d2742938c213 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | tied | better | graph_prior_only | 40 | 34175cd56b59 | 00a235e8a58f |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | tied | better | graph_prior_only | 40 | 457035083107 | e206ba924576 |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | worse | better | graph_prior_only | 100 | da51e84b6771 | ae91df3d52f0 |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | tied | better | graph_prior_only | 40 | 6a199894b017 | 33513c341b71 |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | tied | better | graph_prior_only | 40 | 6a9ab20ffffe | db1459cae874 |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | tied | better | graph_prior_only | 40 | 309cc213c85d | a6cab82391cd |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | worse | better | graph_prior_only | 80 | d103763cd869 | e22a13f0a5ad |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | da75e379a586 | 4f95d224642f |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | tied | better | graph_prior_only | 40 | 785d20128773 | 924b6422d38e |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | tied | better | vector_only | 60 | 4ec9b9f13b52 | 9291f05ae4fd |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | tied | better | graph_prior_only | 40 | 26870769ca40 | bf56d9123f42 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | tied | better | graph_prior_only | 40 | a454fcff645b | 841af0b97fe0 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | tied | better | graph_prior_only | 40 | 7c19720e1c88 | 3b7ef62345f4 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | tied | better | graph_prior_only | 40 | 0244c674109c | 8bc125fb11b2 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | tied | better | graph_prior_only | 40 | 1c109ddd548f | f1ec365244a8 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | tied | better | graph_prior_only | 40 | d979c55f4eb3 | 2f9011f3b741 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | tied | better | graph_prior_only | 40 | f326787dbd13 | 10d1c2ae80ea |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | tied | better | graph_prior_only | 40 | d69367348dd6 | cc5b2a14c9ad |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | tied | better | graph_prior_only | 40 | 8ee1946cf57e | 913e872b6ac8 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | tied | better | graph_prior_only | 40 | d5ebd335d7a3 | fc6ebb6dd952 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | tied | better | graph_prior_only | 40 | 29236086ca7c | a51101650f2a |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | tied | better | graph_prior_only | 40 | e3caec0f4bee | 4ca3e70e5177 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | tied | better | graph_prior_only | 40 | 4a84236be9c6 | 65cd8e0cafc6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | tied | better | graph_prior_only | 40 | df6d529c2598 | a25d3b97f2d6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | tied | better | graph_prior_only | 40 | 53ad958f9a6c | c698ecbe45c4 |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | d5ee1c2c95fe | 294b691e2946 |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | tied | better | vector_only | 80 | 2b1aa1c3708e | 6a65ea4faf62 |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | worse | better | graph_prior_only | 60 | 9fb0ddce2570 | 684b0331dbf8 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | tied | better | graph_prior_only | 40 | ac3571810451 | b95406c85340 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | tied | better | graph_prior_only | 40 | 1fd2b605e7d2 | d178b54dd66d |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | tied | better | graph_prior_only | 40 | 130cf15f9cde | 1509784bccec |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | tied | better | graph_prior_only | 40 | 9c7397c6aab3 | 88f30af716b2 |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | tied | better | graph_prior_only | 40 | 8d18649d079b | 2c7d920c1dd7 |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | tied | better | graph_prior_only | 40 | 29bb679f705f | e8a34b312b17 |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | worse | better | graph_prior_only | 60 | ec0a05f66b40 | d0b1b75aa58a |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | tied | better | graph_prior_only | 40 | 4c3155a3ed27 | 8eec058e861b |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | worse | better | graph_prior_only | 100 | ccc1f606445c | 62465478f20f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | tied | better | vector_only | 70 | 92b5637da04e | 985c00a70ef4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | tied | better | graph_prior_only | 40 | eb3e01458e4a | 37b8bd284ae8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | tied | better | graph_prior_only | 40 | 3e59026ae09f | 4abd7eb7038b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | tied | better | graph_prior_only | 40 | 439c072dcfcc | a6b4925a56f6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | tied | better | graph_prior_only | 40 | a7c145a884c2 | e76b50a13db3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | tied | better | graph_prior_only | 40 | b8dafe73ac5a | e5525a064110 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | tied | better | graph_prior_only | 40 | 811c803213d0 | 40625f2d3576 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | tied | better | graph_prior_only | 40 | d7e47758772e | 2b18dba2b0bf |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | tied | better | graph_prior_only | 40 | 7a5f7db275a9 | 405c40d5b982 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | tied | better | graph_prior_only | 40 | 4691a42b6d68 | d8ea35af40df |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | tied | better | graph_prior_only | 40 | 83be3f951df4 | 43b4a2f86bc7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | tied | better | graph_prior_only | 40 | 7d2aa2e7d427 | c9bb0f0eb36a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | tied | better | vector_only | 100 | ac80a07149a6 | 1efdf2a64fbf |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | worse | better | graph_prior_only | 100 | 74a1d7145c8f | 45dfab8dfa8a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | tied | better | graph_prior_only | 40 | d77667c64f40 | 7848b40b4258 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | tied | better | graph_prior_only | 40 | dc71e111c883 | b5804a1dc2a6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | tied | better | graph_prior_only | 40 | 48ae47f463e4 | 7cc5cb65a7fa |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | tied | better | graph_prior_only | 40 | 82c02d55e70e | 410cde793f26 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | tied | better | graph_prior_only | 40 | eb26f064bc52 | e23b263c8186 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | tied | better | graph_prior_only | 40 | 22aa3e127be3 | b61cf9cf134e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | tied | better | graph_prior_only | 40 | 52a6ecb726dd | 623ed86967b6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | tied | better | graph_prior_only | 40 | 613b1549f0e2 | 5feddde7ab2d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | tied | better | graph_prior_only | 40 | 4ad65663b533 | ad4d83d5ea81 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | tied | better | graph_prior_only | 40 | e6994316d1b5 | 8c0eceba1a6b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | tied | better | graph_prior_only | 40 | dac907f36da0 | 35b97cb64667 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | tied | better | graph_prior_only | 40 | e168ba695985 | 32703d98c7d5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | tied | better | graph_prior_only | 40 | 62e1be59f06f | 11d4ebff01da |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | tied | better | graph_prior_only | 40 | 6ea13ee6108f | 5c00a34f1e52 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | tied | better | graph_prior_only | 40 | 3ff5df3646ff | 326e90888014 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | worse | better | graph_prior_only | 100 | 59aaf857d1ca | 5ff07f0578a7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | tied | better | graph_prior_only | 40 | 6bd1b7be01cc | 2ac052300d3a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | tied | better | graph_prior_only | 40 | 093117f5eaed | 062ffe1873de |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | tied | better | graph_prior_only | 40 | 7ba01d4d8d30 | 1a737ccf95f7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | tied | better | graph_prior_only | 40 | d8496e906792 | c4f42018f311 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | tied | better | graph_prior_only | 40 | f70f7e0adeb2 | 0f5f630ff54b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | tied | better | graph_prior_only | 40 | 9d47b4906a08 | 69a90e650032 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | tied | better | graph_prior_only | 40 | e03d4aa83438 | 1ca570eddee6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | tied | better | graph_prior_only | 40 | bcfb569f4db5 | cc6c04df23ad |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | tied | better | graph_prior_only | 40 | 9cecdad04f8b | 59e191973dec |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | tied | better | graph_prior_only | 40 | 04daf5ce2ea6 | 49249f4d555f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | tied | better | graph_prior_only | 40 | c280fce66f01 | fe69d8ae37df |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | tied | better | vector_only | 100 | c89d82d4651f | f902219c03d2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | tied | better | graph_prior_only | 40 | c07a16d3262a | 36c106c0966c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | tied | better | graph_prior_only | 40 | 5988cc7a08dd | cf414bccca90 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | tied | better | graph_prior_only | 40 | 129751824c0a | bb2bc3c371f4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | tied | better | graph_prior_only | 40 | 104dd47b6197 | 3caa203bf5e2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | tied | better | graph_prior_only | 40 | 16ea523c4a9b | 61ca5698b5d4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | tied | better | graph_prior_only | 40 | 89f17ce75651 | 8ce4ca4987ab |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | tied | better | graph_prior_only | 40 | 577bff3be358 | 29bc838552c5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | tied | better | graph_prior_only | 40 | 5b50748b81b1 | 162070798301 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | tied | better | graph_prior_only | 40 | e74b9f3e369f | 0d3d389e7ee8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | tied | better | graph_prior_only | 40 | af2f58429a48 | 5a05f4ca5cc3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | tied | better | graph_prior_only | 40 | a55a27bbc260 | dd2d35946ff5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | tied | better | graph_prior_only | 40 | b7fc7d93d174 | e6042ec516ce |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | tied | better | graph_prior_only | 40 | 34c965af8631 | 7637fcbeb33f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | tied | better | graph_prior_only | 40 | def6e862c4d8 | 3183aa262309 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | tied | better | graph_prior_only | 40 | 7fba4d76bfce | 2a4eb17e4234 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | tied | better | graph_prior_only | 40 | fe45e2bce561 | cfe3d57efe2d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | tied | better | graph_prior_only | 40 | b12681371229 | 3a637090bfe7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | tied | better | graph_prior_only | 40 | bfd7645a346c | fb536228a65e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | tied | better | graph_prior_only | 40 | 046828935991 | 3faaa4e07e38 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | tied | better | graph_prior_only | 40 | 84bd04b76bbd | 5ae16460a875 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | tied | better | graph_prior_only | 40 | 374271e95263 | c0784fe59075 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | tied | better | graph_prior_only | 40 | a7a874c7c58a | 9eebbd854695 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | tied | better | graph_prior_only | 40 | 051ef0ae6919 | 1a86f489e638 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | tied | better | graph_prior_only | 40 | cf2e1b8c5087 | bf4c8fa7bfdc |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | tied | better | graph_prior_only | 40 | 5bda8a325e2d | be0c1359f127 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | tied | better | graph_prior_only | 40 | a4684424e1c6 | e2819590c89e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | tied | better | graph_prior_only | 40 | f5b16f76fe8f | 10a50dbbf027 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | tied | better | graph_prior_only | 40 | 2b69b3d24eb6 | 235bdacf642d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | tied | better | graph_prior_only | 40 | 901134321415 | 6a3c6f6f1737 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | tied | better | graph_prior_only | 40 | 6ac4a60f9330 | dfbe0fb6e2c0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | tied | better | graph_prior_only | 40 | eb51de16158f | 91166a05bef1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | tied | better | graph_prior_only | 40 | 892e82c7e268 | 9e9a8d5cb744 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | tied | better | graph_prior_only | 40 | 9ce26ef28961 | 6ce7b16a4e2a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | tied | better | graph_prior_only | 40 | 3041272027e7 | 0083a6534a9f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | tied | better | graph_prior_only | 40 | 8f80ad22a605 | 8cb6aff5f0c8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | tied | better | graph_prior_only | 40 | 5d9f29fa6224 | f84eddfa818e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | tied | better | graph_prior_only | 40 | 04b789e7d9e0 | d5d3455fa77a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | tied | better | graph_prior_only | 40 | 51f2b00cd87b | 0a8f53232c0f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | tied | better | graph_prior_only | 40 | 99ced4b2fbd5 | b42b92144a2e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | tied | better | graph_prior_only | 40 | 7678c4f2688c | 4954c609fdbc |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | tied | better | graph_prior_only | 40 | 979695996170 | 6f16dff1b1ad |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | tied | better | graph_prior_only | 40 | f7f2f894e51b | ddab66e349e7 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | tied | better | graph_prior_only | 40 | ea5f9202f4ba | 9edaa3b10a9f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | tied | better | graph_prior_only | 40 | 02da6d134554 | ebf44f1e0edf |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | worse | better | graph_prior_only | 60 | d29dcf605705 | 1aa3f3c9b04c |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | tied | better | graph_prior_only | 60 | c1d8bd5b9aef | fe7f8b01f8fd |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | tied | better | graph_prior_only | 40 | a3a22a758bae | 38bfb0530318 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | tied | better | graph_prior_only | 40 | f5697e01d094 | 015ed4563792 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | tied | better | graph_prior_only | 40 | 190725943fb3 | 64d0e1ef4b07 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | tied | better | graph_prior_only | 40 | 729e91c96e2f | 6f7b3132acb9 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | tied | better | graph_prior_only | 40 | 8168eff97757 | a64e33218216 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | tied | better | graph_prior_only | 40 | a02e6f94d421 | 2779305a6c57 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | tied | better | graph_prior_only | 40 | 2a35017bd98f | 504a5f140418 |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | tied | better | graph_prior_only | 40 | b0babfb72920 | 6b29f372c3d8 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | worse | better | graph_prior_only | 60 | 23b3bfb781da | 42f8347dec13 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | tied | better | graph_prior_only | 40 | acd51dc37189 | 5b4ef8f274fd |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | tied | better | graph_prior_only | 40 | baa001df8ebf | 8c43a3ab7802 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | worse | better | graph_prior_only | 60 | e64f44c7ba93 | 21d801e7df15 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | tied | better | graph_prior_only | 40 | 06aab0ca17bb | 7372b7a22909 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | tied | better | graph_prior_only | 40 | ce2acd5c0fd0 | cb2e4bc849c2 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | tied | better | graph_prior_only | 40 | 695b72c3d045 | 0ae838ec684b |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | tied | better | graph_prior_only | 40 | 8e93bbb3c9f8 | 66c7e4fbc253 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | tied | better | graph_prior_only | 40 | 403002c4788a | 5a028cf4fb0d |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | tied | better | graph_prior_only | 40 | 54a95d52a81a | 7309b9e05016 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | tied | better | graph_prior_only | 40 | be3529533127 | 0cb637b60338 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | tied | better | graph_prior_only | 40 | 7f89ce9daaab | 0881e580576b |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | tied | better | graph_prior_only | 40 | 654014a65338 | bf1f1b1a43d9 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | tied | better | graph_prior_only | 40 | 31042189ae1b | 9e404c10ffe4 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | tied | better | graph_prior_only | 40 | 1c41ac14f2aa | bbacf82a33ca |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | tied | better | graph_prior_only | 40 | bf533c6c0b24 | 32c87e5eda58 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | tied | better | graph_prior_only | 40 | e273c83397b5 | d4721d163d49 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | tied | better | graph_prior_only | 40 | 2520b25ef85b | 6de45d24916c |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | tied | better | graph_prior_only | 40 | 9f4cc718843d | 041d65ff3a98 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | tied | better | graph_prior_only | 40 | b79819f9bb04 | 037831b2c227 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | tied | better | graph_prior_only | 40 | ca9f4bc54526 | fe72dcd55ea7 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | tied | better | graph_prior_only | 40 | 0cc67514f8d0 | 9fcdde23a8bc |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | tied | better | graph_prior_only | 40 | 097cade7ab99 | 9ceb3a4b44ff |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | tied | better | graph_prior_only | 40 | 14de9424fb08 | eed3f1c5b82f |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | better | better | learned_route | 60 | b40d317b55f4 | 4ef5dbbca5b7 |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | tied | better | graph_prior_only | 40 | 0cc5cf8e954b | 2ceca027aff3 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | tied | better | graph_prior_only | 40 | a7e30a26c687 | 4bf872dcb4d7 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | tied | better | vector_only | 70 | 08d9c6bcff0e | e41fe28f8723 |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | tied | better | graph_prior_only | 40 | 517679ddfbf3 | 071bc22e4692 |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | worse | better | graph_prior_only | 60 | 95b69f06ceed | a062c91e7ccc |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | tied | better | graph_prior_only | 40 | 64a6b7a8ca5c | f45d84dc0407 |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | tied | better | graph_prior_only | 40 | eea68bee7002 | 3e57d370b293 |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | tied | better | graph_prior_only | 40 | 60a1e65e373d | de19ab130a87 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | tied | better | graph_prior_only | 40 | 04738c29cc13 | 2cc02f347f64 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | tied | better | graph_prior_only | 40 | fb06fa388b5b | 390d0cbf2d0a |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | tied | better | graph_prior_only | 40 | 0fa9efe6bb78 | 7f6ebb5c7fb3 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | tied | better | graph_prior_only | 40 | c946ed930842 | 1ee08a70b95e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | tied | better | graph_prior_only | 40 | f310f3e6e9fc | 932a4fccec49 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | tied | better | graph_prior_only | 40 | 73338b990382 | 81bfa41fb8f5 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | tied | better | graph_prior_only | 40 | 83d1304c19b5 | 44c6a26cba70 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | tied | better | graph_prior_only | 60 | 977d862f14f8 | b7ed540fe910 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | tied | better | graph_prior_only | 60 | 29fea1b5f589 | d2513ea54cb0 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | tied | better | graph_prior_only | 40 | 257b385e185e | 9f343dfb67c4 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | tied | better | graph_prior_only | 40 | 4a3a41943d07 | de2f6e891ec2 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | tied | better | graph_prior_only | 40 | 509b60ca0383 | 04ca7aebd9b4 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | tied | better | graph_prior_only | 40 | 3d50de2a0af4 | 0d8d58652e1e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | tied | better | graph_prior_only | 40 | 91cc0713768b | eef5a047776d |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | eb0508b1b0d6 | be1c8309f4f8 |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | tied | better | graph_prior_only | 40 | 0ba09a4feb1c | 4a4fa6c26b47 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | tied | better | graph_prior_only | 40 | 96e3561e0e63 | e143e1dba938 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | tied | better | graph_prior_only | 40 | 4dbd90bcb744 | 3c8600eef198 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | tied | better | graph_prior_only | 40 | 44e3d4ecf1f6 | e2e83f327033 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | tied | better | graph_prior_only | 40 | 800cc8130a63 | 77cd2ff3417b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | tied | better | graph_prior_only | 40 | f5614fd48b92 | 3ea6569a7a61 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | tied | better | graph_prior_only | 40 | df53294dc131 | 68a7456dae5f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | tied | better | graph_prior_only | 40 | 43af858a142b | 4f3f7ff79f54 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | tied | better | graph_prior_only | 40 | 3e2619fc34a5 | ed3f9a9a1115 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | tied | better | graph_prior_only | 40 | d3384b4f4630 | 85816e620111 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | tied | better | graph_prior_only | 40 | a4f49b05a0bf | d7c643c0d17e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | tied | better | graph_prior_only | 40 | 1107e1946d5a | 5ff012bd09bb |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | tied | better | graph_prior_only | 40 | 63a0df346bd4 | 0fc5a7f0d618 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | tied | better | graph_prior_only | 40 | 79e2063f10dd | abafd0dc3ea8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | tied | better | graph_prior_only | 40 | 7324d952842a | c034ead71957 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | tied | better | graph_prior_only | 40 | cf3ae0cfc2c5 | 56d29063d942 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | tied | better | graph_prior_only | 40 | 1fd6ef968d55 | 5b4f2b202797 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | tied | better | graph_prior_only | 40 | 6bb0cf92a829 | d419ae00d815 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | tied | better | graph_prior_only | 40 | a9c2d65a4a1b | 86b48e975d53 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | tied | better | graph_prior_only | 60 | 534b5edf2c05 | cdd3fa2919b7 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | tied | better | graph_prior_only | 40 | bf2f77ff7b4a | aaa1a26f6794 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | tied | better | graph_prior_only | 40 | d815f246a131 | fdd89d7e7ec9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | tied | better | graph_prior_only | 40 | 724eec5618b0 | dd57b8427ddc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | tied | better | graph_prior_only | 40 | 7844248e8ed4 | 61c5ed462351 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | tied | better | graph_prior_only | 40 | 84f781075922 | c0cb767ae01b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | tied | better | graph_prior_only | 40 | f0b82c1300b1 | 71ada67fbb7c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | tied | better | graph_prior_only | 40 | d3d1988a2427 | 3969b8cd608a |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | tied | better | graph_prior_only | 40 | 015abb8b2dbf | 5ac90a394fc5 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | tied | better | graph_prior_only | 40 | fc2fd44b45c0 | c669370cbe36 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | tied | better | graph_prior_only | 40 | c29d063fe2ef | af8fb41cb44e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | tied | better | graph_prior_only | 40 | 858ad6d6bda4 | 29568cc2719b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | tied | better | graph_prior_only | 40 | d1cfe5625649 | 2dc97035d581 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | tied | better | graph_prior_only | 80 | 1c5ae0f7f07d | 1c48f17d9fe6 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | tied | better | graph_prior_only | 80 | 695d2154b5b1 | 35a8988b6e57 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | tied | better | graph_prior_only | 40 | b9ad3efe60c9 | 17867af25912 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | tied | better | graph_prior_only | 80 | 08edc13143d3 | 17f79bd34f3d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | tied | better | graph_prior_only | 40 | 9701011bb271 | 8c3502734621 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | tied | better | graph_prior_only | 40 | fb8c410922e4 | d378c5b93f08 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | tied | better | graph_prior_only | 40 | 51099d3abc15 | 55c2f65a7336 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | tied | better | graph_prior_only | 40 | 219fe10ac678 | 6cc236caaf52 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | tied | better | graph_prior_only | 40 | da4a1b6daff4 | 2ad3d4ae1760 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | tied | better | graph_prior_only | 40 | b66c1ccb3890 | deb7978e5690 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | tied | better | graph_prior_only | 40 | 370f640a58a1 | 6eff3105f00f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | tied | better | graph_prior_only | 40 | 1a622806aafa | 49cc579e341e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | tied | better | graph_prior_only | 40 | 70873112a322 | ad87e8b35123 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | tied | better | graph_prior_only | 40 | 549960008043 | 9e2ba734958c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | tied | better | graph_prior_only | 40 | 56a505c34987 | 15c007e352a0 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | worse | better | graph_prior_only | 60 | c76f5f4d759b | a001b5c87002 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | tied | better | graph_prior_only | 40 | 560336344027 | 43ad12b6d256 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | worse | better | graph_prior_only | 60 | dfd198e7a254 | 7e4463ff3d49 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | tied | better | graph_prior_only | 40 | 5be5918ce289 | 1a2fef9cf386 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | tied | better | graph_prior_only | 40 | 94155f43586c | 8013245842de |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | worse | better | graph_prior_only | 70 | 2db057f016e7 | 1c5ed582bc28 |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | tied | better | graph_prior_only | 40 | f78442b6cea3 | d533aa827396 |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | tied | better | graph_prior_only | 40 | 66215fc39548 | 75b132223885 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | tied | better | graph_prior_only | 40 | a265fc5df6e1 | ac6e4fbbdccc |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | tied | better | graph_prior_only | 40 | 91e298f55cb2 | 27f39138e236 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | tied | better | graph_prior_only | 60 | a7aebb76bbbe | 5793a982c8fe |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | tied | better | graph_prior_only | 40 | 26d56e6aee08 | 2704a72775f7 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | tied | better | graph_prior_only | 40 | 936d11602fb3 | 0e5b36706d1d |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | tied | better | vector_only | 60 | 84124b50c144 | 7219314450ed |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | tied | better | graph_prior_only | 40 | 24862da47ac6 | d3fd9500efa4 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | worse | better | graph_prior_only | 70 | da0ef73a2b11 | 636763ddcb19 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | tied | better | graph_prior_only | 40 | 7e99aaa9c0e8 | b9725c28f920 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | tied | better | graph_prior_only | 40 | 819f3a152bce | 4e13af746fc0 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | tied | better | graph_prior_only | 40 | deac5883d1b4 | 272648cd532a |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | tied | better | graph_prior_only | 40 | 9375e05147cb | a7807cab98cd |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | tied | better | graph_prior_only | 40 | 8ba438b7eb5a | 76f6117753e9 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | tied | better | graph_prior_only | 40 | 40627afb3909 | 880dff546c96 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | tied | better | graph_prior_only | 40 | 2b93740e136c | 633aa7766410 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | tied | better | graph_prior_only | 40 | 1e5f61a6c322 | aacaa1338b18 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | tied | better | graph_prior_only | 40 | 0efbc1bf9f2c | a8452211b50b |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | tied | better | graph_prior_only | 40 | f9b023bdcdcc | 1aca0906fe44 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | tied | better | graph_prior_only | 40 | be6cf4c5414d | b9dbd9eded95 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | worse | better | graph_prior_only | 70 | f7b7be929995 | b5f469560e35 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | tied | better | graph_prior_only | 40 | 3e7e351c7d50 | bad98d89ed20 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | tied | better | graph_prior_only | 40 | 54d2245b8fae | 65ad3e6fe16f |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | tied | better | graph_prior_only | 40 | 15ae3da3c468 | 43c3558bce7a |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | tied | better | graph_prior_only | 40 | 32d38b3e71d0 | ab4b5705cbb8 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | tied | better | graph_prior_only | 40 | f2b3804ab34d | 7fa5e122b988 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | tied | better | graph_prior_only | 40 | f53be3aea741 | e5fe5b389ec2 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | tied | better | graph_prior_only | 40 | bca81685eca6 | 26dd94e02a39 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | tied | better | graph_prior_only | 40 | f33838b0ccda | 1f6dd96791d5 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | tied | better | graph_prior_only | 40 | 9021bef4a304 | 675270d709e3 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | tied | better | graph_prior_only | 40 | c78e5fb2c143 | 3d215c5db007 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | tied | better | graph_prior_only | 40 | fb895be9279f | f66e5d88b34c |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | tied | better | graph_prior_only | 40 | 678da907ec6b | ff4f3caf725e |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | tied | better | graph_prior_only | 40 | c738c76e6d5a | 34e071b9144d |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | tied | better | graph_prior_only | 40 | 225959be1050 | d6e161ce4317 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | tied | better | graph_prior_only | 40 | 7f16d9997249 | 2dff8c07aeae |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | tied | better | graph_prior_only | 40 | 4dfde11fc749 | 44ed371aafc1 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | tied | better | graph_prior_only | 40 | 614315e2b9e8 | 7b492971235f |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | tied | better | graph_prior_only | 40 | 61e77ec469fe | 6803fc5d76aa |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | tied | better | graph_prior_only | 40 | 261c0d4e7bee | 5e9a8cbdd4b9 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | tied | better | graph_prior_only | 40 | 01f62d51b959 | 03d34778f56d |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | tied | better | graph_prior_only | 40 | 61f64cd38ba3 | f6bef4bd2f53 |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | tied | better | graph_prior_only | 40 | 1ebf73384a54 | 9966ff062c35 |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | tied | better | graph_prior_only | 40 | 7fd80eaa87d8 | 3fb1e38b8fa5 |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-7f1662035536442d6dfd9aacb23bf9fc38a2d8aaa36a03cf24f34d5f389d3c07 |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-67a1709848831208d0b97c891c824cd46a70b94e931286c4c1c34de8fae84845 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-ad3bb5ed70979aae56a8fdc2455c425c9b729798fd6a5e3ef8cdf72194080c00 |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-2b09e80a42fef271a0948f79bc43152e60db93be6aa8329b1a6701daad609887 |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ff9a5d7fa04a6ea40a5b485e20bfab28357447ca80fcda5e750a35c8fdaa5d83 |
| worked-traces | worked-traces.md | none | sha256-87f50a027fd58a689fa6da4dea17740fbbf55ebd9e7960f4f4b9a30bc1882f2e |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-0a67cc863a6b2ca4c5ae273e9a2863687a8fefbc31da452b0ad102c542d26c40 |
