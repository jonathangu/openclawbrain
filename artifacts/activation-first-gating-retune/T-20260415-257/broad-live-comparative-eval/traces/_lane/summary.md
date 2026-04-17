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
- success-adjusted economics: learned_route used 169 estimated prompt tokens, 0.000211 estimated prompt USD, and 10 ms serve-path latency per incremental win vs graph_prior_only 293, 0.000366, and 10
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
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | worse | better | graph_prior_only | 60 | 2cfa2d806ad1 | 15cf5d5481a7 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | tied | better | graph_prior_only | 40 | 6e3d22c977d1 | 049df50f42f2 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | worse | better | graph_prior_only | 70 | dc5e7461987f | e646d7e8876b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | worse | better | graph_prior_only | 80 | 8898ea296ee5 | 5e5224ac0590 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | tied | better | graph_prior_only | 40 | 49609a58beea | 509d2e12ba5c |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | tied | better | graph_prior_only | 60 | bb12d75a4377 | 49bd796ebd7f |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | worse | better | graph_prior_only | 60 | 4ccb85c9c812 | bb132b58610b |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | worse | better | graph_prior_only | 60 | fa11f655fcdd | 1d44a666e324 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | tied | better | graph_prior_only | 60 | bf02146a8499 | 8b380d383a63 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | tied | better | graph_prior_only | 40 | 3c7336884853 | d74b7701a6de |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | tied | better | graph_prior_only | 60 | 21553f9423a8 | 9a1ae83d37f0 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | tied | better | graph_prior_only | 40 | bb36b4058d46 | 94659a6b519e |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | tied | better | graph_prior_only | 40 | 6e7a73924a37 | 9a6217d06279 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | tied | better | graph_prior_only | 40 | 49276bd0d5cf | 0b38dcff402a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | tied | better | graph_prior_only | 40 | ddb69b1c396b | 55499d36fa82 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | tied | better | graph_prior_only | 40 | 9243bbf175f0 | 2de7f3521cc3 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | worse | better | graph_prior_only | 60 | 8253d553ba13 | 183503b03f0c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | tied | better | graph_prior_only | 40 | af80c8ca0d5f | 09ae2f2a9837 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | tied | better | graph_prior_only | 40 | c4c07d9d3ad4 | 437087b75a55 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | tied | better | graph_prior_only | 40 | 664f2d7130d8 | f681adf18540 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | tied | better | graph_prior_only | 40 | cd94fc2c093c | 83dc4a86ec41 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | worse | better | graph_prior_only | 100 | 3b925b7233a4 | 8e14d4e322ad |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | tied | better | graph_prior_only | 40 | 49f8e47c8dfd | 6e2bef9763cc |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | tied | better | graph_prior_only | 40 | 59fed24cd007 | 0d423139a0c9 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | worse | better | graph_prior_only | 100 | 38d53ffa0672 | f291af94c0b5 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | tied | better | graph_prior_only | 40 | 39ce4e21772e | 2d86ceb09220 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | tied | better | graph_prior_only | 40 | 86c5ef89a5d4 | 52a152938ef6 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | worse | better | graph_prior_only | 100 | f6a05d3a6fde | 4b727feb181e |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | tied | better | graph_prior_only | 40 | be8515f9bcc6 | ce7c2f86de05 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | tied | better | graph_prior_only | 40 | 1192f14883d1 | 35bd4a45fb63 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | tied | better | graph_prior_only | 40 | a3c790b61535 | fcc579169791 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | tied | better | graph_prior_only | 40 | 8a4780a6228c | 02c853501e7a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | worse | better | graph_prior_only | 60 | 6e0625333f5d | 2de17bdd5d5b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | b38b3ea1c426 | 90357b6a871a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | tied | better | graph_prior_only | 40 | f02a17a2f510 | f80d7020b4ab |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | tied | better | graph_prior_only | 40 | 972854a1edaa | 909d173ae2ae |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | tied | better | graph_prior_only | 40 | 3b915e9639aa | 008d816e264e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | tied | better | graph_prior_only | 40 | 0849eaa18a84 | 2a650637428e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | tied | better | graph_prior_only | 40 | d4c8d50d9b0d | d1c6cc4b0a98 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | tied | better | graph_prior_only | 40 | ee3bf8ea2605 | ba39769bf96b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | tied | better | graph_prior_only | 40 | 87a0cf80f165 | e2e07f06af21 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | tied | better | graph_prior_only | 40 | 46d528bd9d92 | d19916ef8346 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | tied | better | graph_prior_only | 40 | 89f62d30b45e | 24fdcf30b7d5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | tied | better | graph_prior_only | 40 | f5a40c92b461 | e6168c244180 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | tied | better | graph_prior_only | 40 | f55f53ddd4fd | a55b3a78233e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | tied | better | graph_prior_only | 40 | e2cec65d47c2 | b1c2a150c2bc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | tied | better | graph_prior_only | 40 | 96864e0420b8 | 1cc92a2ab08d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | tied | better | graph_prior_only | 40 | 68c9ed99cf16 | c8be769772ca |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | tied | better | graph_prior_only | 40 | f31ded63cb07 | 4a369615bec6 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | tied | better | graph_prior_only | 40 | e0a5a141555a | ed8bb07b3a7b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | tied | better | graph_prior_only | 60 | d82dbc6f6d93 | 210f15ab0f6a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | tied | better | graph_prior_only | 40 | aaacccb82d5f | 88a61f07c248 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | tied | better | graph_prior_only | 40 | a34ad34d6254 | d8c8701a2aec |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | worse | better | graph_prior_only | 60 | d17cc21a60f1 | 267bb6d2b2d7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | tied | better | graph_prior_only | 40 | b831982c4822 | 048a503cc824 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | tied | better | graph_prior_only | 40 | c0489fe1af0f | 0d4653d49f26 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | tied | better | graph_prior_only | 40 | f0ec16de135c | f67777629841 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | tied | better | graph_prior_only | 40 | ccdefd47d81d | 6f358d8cb7f4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | tied | better | graph_prior_only | 40 | b149bbdb0dfb | 63428f7cb7ec |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | tied | better | graph_prior_only | 40 | 3763ad869de5 | ebecd77f0096 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | tied | better | graph_prior_only | 40 | a1eaa52b3e7f | b4352106fa73 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | tied | better | graph_prior_only | 40 | 8650518a1ddd | ef2db2cd9a87 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | tied | better | graph_prior_only | 40 | 1933bdf8841d | 7bb7131d299e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | tied | better | graph_prior_only | 40 | 5f378a0b83d9 | 04d85c3158e9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | tied | better | graph_prior_only | 40 | b17aa0520a30 | 13d6836106a9 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | tied | better | graph_prior_only | 40 | 2fb91d754e78 | 4c34cdb641ef |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | tied | better | graph_prior_only | 40 | 51256b867a92 | b61fe118a1d2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | tied | better | graph_prior_only | 40 | 6fd711ee270b | 587d1ecb7b8d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | tied | better | graph_prior_only | 60 | 528d7ea807a9 | a8eaafe0429e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | tied | better | graph_prior_only | 40 | b15d3840c987 | 8eaa7fbcb4de |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | tied | better | graph_prior_only | 40 | 16a547628fb4 | 5bf9b0d98d41 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | tied | better | graph_prior_only | 40 | 4c2ef81b9937 | f9497ae0ea47 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | tied | better | graph_prior_only | 40 | 87acac7fc785 | 3761f78b6534 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | tied | better | graph_prior_only | 40 | df286c0f6a2b | 03a1f1dceaad |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | tied | better | graph_prior_only | 40 | 0946cc1e18d5 | d45870d79a2f |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | tied | better | graph_prior_only | 40 | 037f706bfda3 | 38bdb46975d0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | tied | better | graph_prior_only | 40 | 4fd143b4dd6c | 9f44831f263e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | worse | better | graph_prior_only | 60 | ee70f0e2e199 | 3faccddaee2a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | tied | better | graph_prior_only | 40 | a9a151fc8f69 | d3286363e67a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | tied | better | graph_prior_only | 40 | 99e2a46df3e6 | 5ba0b529152a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | tied | better | graph_prior_only | 40 | 04889e2d7db1 | 8fd749aa6701 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | tied | better | graph_prior_only | 40 | 87b18b2a7833 | df2b68c28abe |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | tied | better | graph_prior_only | 40 | e7d37640a6e4 | c1e6137eb5c4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | tied | better | graph_prior_only | 40 | 2fc35d7cf931 | 6412f1b8568a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | tied | better | graph_prior_only | 40 | aafda7ce29bf | 999868909f3e |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | tied | better | graph_prior_only | 40 | 6d789fc4d54a | 75669052ea9a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | tied | better | graph_prior_only | 40 | afd5ebf3e131 | 9fbe0424afc5 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | tied | better | graph_prior_only | 40 | c4439bb59657 | 7b7b93d5cb65 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | tied | better | graph_prior_only | 40 | be4c6bd90a9e | 0817c00785eb |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | tied | better | graph_prior_only | 40 | d7699438182d | cd23a2a8b9c1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | tied | better | graph_prior_only | 40 | 30aae78bc478 | d8da044eb3f8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | tied | better | graph_prior_only | 40 | f9e779db889f | 51c7e05d1ed0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | tied | better | graph_prior_only | 40 | 8437d0a2fdb7 | b218ade7730d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | tied | better | graph_prior_only | 40 | 0bb85a7d97e3 | b16de6f274d8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | tied | better | graph_prior_only | 70 | 69b63a634f03 | 1a9287223eab |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | tied | better | graph_prior_only | 40 | 75625b275ffa | 716a99bb3c1f |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | tied | better | graph_prior_only | 40 | 3120071236f2 | d6e744170190 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | tied | better | graph_prior_only | 40 | cea0b144be0b | 19feeff0d9fc |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | tied | better | graph_prior_only | 40 | 4ccd571d2cfe | ad674c493c0f |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | tied | better | graph_prior_only | 40 | 6dfb88c95eb2 | d495f13c047c |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | tied | better | graph_prior_only | 40 | 5481cc33bfd1 | dd5c3a64cb52 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | tied | better | graph_prior_only | 40 | 8423f104b8c0 | e4f2496f2777 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | tied | better | graph_prior_only | 40 | 6761a1fe1aaa | dcad238bcb08 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | tied | better | graph_prior_only | 40 | 58a6a19811fa | 2def7225f83d |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | tied | better | graph_prior_only | 40 | e100249f9988 | 4c810c9a186b |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | tied | better | graph_prior_only | 40 | 44f42910b1e4 | a7cbfc61c144 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | tied | better | graph_prior_only | 40 | 89044dbf7780 | 506f540e89ce |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | tied | better | graph_prior_only | 40 | d0f36c076665 | a0621fa057a3 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | tied | better | graph_prior_only | 40 | dd9a8b2202fe | 9a3f17819272 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | tied | better | graph_prior_only | 40 | e303292f9e95 | 0c1d2070c1d6 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | worse | better | graph_prior_only | 60 | 45c0026d0c34 | c7830649c10d |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | tied | better | graph_prior_only | 40 | b322f3946d75 | 4e5ec02e6b68 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | tied | better | graph_prior_only | 40 | 01f886e98027 | ec6e8b33b766 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | tied | better | graph_prior_only | 40 | 8bc7287b19d4 | 022caf7dbe13 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | worse | better | graph_prior_only | 100 | bc7aab4a6acc | 1e0b9277137c |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | tied | better | graph_prior_only | 40 | 50d13ea46811 | 5677bb90cac8 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | tied | better | graph_prior_only | 40 | f74bd603f81f | ba6a8213193c |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | tied | better | graph_prior_only | 40 | ff8c97e0e83e | a622c22298ec |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | tied | better | graph_prior_only | 70 | 539c4a0d6600 | 4b9846f08891 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | tied | better | graph_prior_only | 40 | f8f78bb12f19 | 8268e8f590f9 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | tied | better | graph_prior_only | 40 | fc8a096b7656 | daa33196cf1a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | tied | better | graph_prior_only | 40 | 34e503a570f7 | a3c4cc93084a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | tied | better | graph_prior_only | 40 | 7d8388b5c398 | 4d5550987923 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | tied | better | graph_prior_only | 40 | 3f2011829b74 | d2742938c213 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | tied | better | graph_prior_only | 40 | f96ce8b280d7 | 00a235e8a58f |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | tied | better | graph_prior_only | 40 | 4fb17bfcd663 | e206ba924576 |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | worse | better | graph_prior_only | 100 | fd7d52620324 | 0b5564ee9c29 |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | tied | better | graph_prior_only | 40 | a8a287120731 | 33513c341b71 |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | tied | better | graph_prior_only | 40 | ca06c6d26677 | db1459cae874 |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | tied | better | graph_prior_only | 40 | acdd23ff0dcb | 10e3e687f986 |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | worse | better | graph_prior_only | 80 | da414a1dfcb7 | 45a5889e3ef9 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | dac0d7fe8aca | 932f285af25e |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | tied | better | graph_prior_only | 40 | d84036d65751 | d9ea89481233 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | tied | better | vector_only | 60 | a8c751827af7 | 1bf97dcb5f81 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | tied | better | graph_prior_only | 40 | 6e5bcfe36aa6 | 0a1048ab7cc6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | tied | better | graph_prior_only | 40 | 6930bc40b35a | 87ab60fad57e |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | tied | better | graph_prior_only | 40 | 064003ab938a | 02e4e40a0f37 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | tied | better | graph_prior_only | 40 | 7363d0666143 | 606a3230c3c8 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | tied | better | graph_prior_only | 40 | 155323051d72 | e3a0cb7afa0b |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | tied | better | graph_prior_only | 40 | d20eb2b4d4f3 | 080e8ccf390c |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | tied | better | graph_prior_only | 40 | 826f2e3e2845 | 1b539f219cc8 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | tied | better | graph_prior_only | 40 | 04e777b9cd84 | e5c834cd1753 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | tied | better | graph_prior_only | 40 | 0b433fa0d4dc | 93efc97a22d4 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | tied | better | graph_prior_only | 40 | ca7d17d6816d | cc514f947947 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | tied | better | graph_prior_only | 40 | 5edb7ead2e40 | f0342feaeca3 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | tied | better | graph_prior_only | 40 | db27ca222a02 | b22c9c350bc6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | tied | better | graph_prior_only | 40 | 7b07e18692db | 938883f66037 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | tied | better | graph_prior_only | 40 | 682ed8f84f3e | 05da8306920b |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | tied | better | graph_prior_only | 40 | c2cad8ad5225 | 57cd9e2949c8 |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 9a805e6646c3 | 4ad59a79174a |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | tied | better | vector_only | 80 | 79dacc978ba2 | efba7dd62607 |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | worse | better | graph_prior_only | 60 | e51b45e5679c | d2ac74d40fa3 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | tied | better | graph_prior_only | 40 | 0d67b92a2a2f | b91b7741d6db |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | tied | better | graph_prior_only | 40 | 938fe22d08aa | f7856c67dbe7 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | tied | better | graph_prior_only | 40 | 7e6ad5828ea4 | ad23073f0bd0 |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | tied | better | graph_prior_only | 40 | 54f78d1c209f | f66f74cca128 |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | tied | better | graph_prior_only | 40 | c2dfcad3324d | dc1d615a110c |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | tied | better | graph_prior_only | 40 | 9480a27d4269 | 4fc48fbed8ff |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | worse | better | graph_prior_only | 60 | f4ac893a2362 | da349d29ea2d |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | tied | better | graph_prior_only | 40 | 0de125cba4d8 | 8eec058e861b |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | worse | better | graph_prior_only | 100 | af8c782a215d | 7890c1d1ae94 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | tied | better | vector_only | 70 | be4912a0137a | 5703a3d308af |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | tied | better | graph_prior_only | 40 | 62d60f2131e9 | c0a000c7eb23 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | tied | better | graph_prior_only | 40 | 62a551bb1552 | 94b4eef9b991 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | tied | better | graph_prior_only | 40 | 42a91509ee1e | a45492567802 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | tied | better | graph_prior_only | 40 | b1877aa88a6c | 578f1b57f3f2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | tied | better | graph_prior_only | 40 | e0bb93f8355d | 237cd558fa7f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | tied | better | graph_prior_only | 40 | cd11d30c028d | c2a4e4c1b891 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | tied | better | graph_prior_only | 40 | 237e43884d94 | 080364a2d0c1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | tied | better | graph_prior_only | 40 | 2cccefce22da | 82e87146a030 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | tied | better | graph_prior_only | 40 | cad27520df7f | cf6db931eef7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | tied | better | graph_prior_only | 40 | ee28a73fc138 | 820b30fffeb7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | tied | better | graph_prior_only | 40 | 0437fb684287 | 74b7aae9faf3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | tied | better | vector_only | 100 | 2bf18fda7d7c | db06814b015d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | worse | better | graph_prior_only | 100 | a9a19808c29a | 7482d6fb5043 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | tied | better | graph_prior_only | 40 | a456de813c7f | 6baf8b74968a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | tied | better | graph_prior_only | 40 | 14ca13fd5027 | 08aaae63f33c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | tied | better | graph_prior_only | 40 | 949faaf22ccd | 699f1d4cab11 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | tied | better | graph_prior_only | 40 | 2620b3c8e30e | c44059eb31f2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | tied | better | graph_prior_only | 40 | 01235afe4b18 | 09dff30641b7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | tied | better | graph_prior_only | 40 | df6fb8c9c930 | b4987fe3dfb1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | tied | better | graph_prior_only | 40 | bf2503dfd59c | b5a7baedb46a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | tied | better | graph_prior_only | 40 | 69fccd758662 | ada75b00e3ed |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | tied | better | graph_prior_only | 40 | b7cbd9c57e3c | 9e27b41fbde2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | tied | better | graph_prior_only | 40 | 2b4944953692 | 0770816edaa0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | tied | better | graph_prior_only | 40 | 085b3d5db201 | c122d5e4330f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | tied | better | graph_prior_only | 40 | 9ef1032b2c33 | 3f12078a8c67 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | tied | better | graph_prior_only | 40 | 419c42bb6c94 | afd33274b534 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | tied | better | graph_prior_only | 40 | 714e33131ae7 | 1e5ac9739051 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | tied | better | graph_prior_only | 40 | 6a88bcd1f0c1 | 51ec2c245c79 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | worse | better | graph_prior_only | 100 | 25ff047ea84b | f34650b22a10 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | tied | better | graph_prior_only | 40 | 52f3cd73029d | 50c24737a6ea |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | tied | better | graph_prior_only | 40 | e78e17bb5abb | 013db56534eb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | tied | better | graph_prior_only | 40 | 2ad4b204c4ab | d83c8ca1cc62 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | tied | better | graph_prior_only | 40 | 8ce6c7cdc3e0 | dbc0f709702e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | tied | better | graph_prior_only | 40 | 6109be8c7b9d | bf952a92f8b0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | tied | better | graph_prior_only | 40 | a2b1d619994a | 6a53eb616e76 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | tied | better | graph_prior_only | 40 | d075ce5ccccc | a02e814dba00 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | tied | better | graph_prior_only | 40 | 0b0d814025e7 | 622b8bb5a683 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | tied | better | graph_prior_only | 40 | cdd0aa9e1ce5 | f4ba616f16f3 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | tied | better | graph_prior_only | 40 | 0c21ca280b04 | 61a3f256d067 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | tied | better | graph_prior_only | 40 | c52ce5aa518c | 125dfd889ec8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | tied | better | vector_only | 100 | 63078c1aa0c0 | bacdc4d92940 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | tied | better | graph_prior_only | 40 | 78bf8f31752b | 48561202519f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | tied | better | graph_prior_only | 40 | 76296ef21811 | 7c317ccf781e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | tied | better | graph_prior_only | 40 | 49e573faa901 | 66e2f3967b72 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | tied | better | graph_prior_only | 40 | 06d0e4da5f5b | 7c488544f05d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | tied | better | graph_prior_only | 40 | 75ea7d7b43fc | 80fa6fd1f284 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | tied | better | graph_prior_only | 40 | 1b0dbd067eb3 | 2314ce0648fa |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | tied | better | graph_prior_only | 40 | 734f9be21cd5 | c59007e4ba87 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | tied | better | graph_prior_only | 40 | 45fc90ac4e7a | 3db7663c64dc |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | tied | better | graph_prior_only | 40 | 2cd387c4d424 | 697611445386 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | tied | better | graph_prior_only | 40 | b3f965603e3c | e9fac6551166 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | tied | better | graph_prior_only | 40 | 9046e5a82d0c | 0d02f4a50433 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | tied | better | graph_prior_only | 40 | 7e1e45c9ae33 | 6fbcb98153c0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | tied | better | graph_prior_only | 40 | 5fde050844f2 | 1a220fa795e6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | tied | better | graph_prior_only | 40 | 3836619e9c3b | 8158022177f2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | tied | better | graph_prior_only | 40 | f254c1f38158 | 598ba9ddf73d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | tied | better | graph_prior_only | 40 | 61012e119232 | eed74950b6b5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | tied | better | graph_prior_only | 40 | 4e54e47c4fbd | 3f6589bedee9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | tied | better | graph_prior_only | 40 | 0dcd70a60e37 | e4fcbd3cf848 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | tied | better | graph_prior_only | 40 | 2350ff4da843 | b41e1a2e80b1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | tied | better | graph_prior_only | 40 | 2a3268d775fd | bc44e719a0d0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | tied | better | graph_prior_only | 40 | 22b7aa6f2d7a | cd85cd4f6a86 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | tied | better | graph_prior_only | 40 | 0abe67f56f58 | 95c282beeeb9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | tied | better | graph_prior_only | 40 | da7aeec57364 | e9abc3c080dd |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | tied | better | graph_prior_only | 40 | 2243f812029a | 6509403fdfa8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | tied | better | graph_prior_only | 40 | 01c2053d18ec | 24f38802f0fe |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | tied | better | graph_prior_only | 40 | cade90df9d7d | cca7f7bfbf35 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | tied | better | graph_prior_only | 40 | dbcc7b5dc80d | ea69df840a96 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | tied | better | graph_prior_only | 40 | 820b6e78f99e | 55840b90788b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | tied | better | graph_prior_only | 40 | 1c82e000f052 | 6e438c571cc9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | tied | better | graph_prior_only | 40 | cb05f3e7ac13 | e126375df2c4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | tied | better | graph_prior_only | 40 | 8ced841c7d79 | ef390b68a30f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | tied | better | graph_prior_only | 40 | e704693ce589 | 5fb50e9d6566 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | tied | better | graph_prior_only | 40 | 578ef571eb9e | 8bff3397d968 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | tied | better | graph_prior_only | 40 | abdccbab74c8 | 9dc1acc18005 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | tied | better | graph_prior_only | 40 | 805396edc188 | 78c3ed9599d7 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | tied | better | graph_prior_only | 40 | c86a4670e849 | b49abbb000f6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | tied | better | graph_prior_only | 40 | c1c8670fe489 | 0dffffa6258c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | tied | better | graph_prior_only | 40 | ea4f0f746fe2 | dd7e20108e69 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | tied | better | graph_prior_only | 40 | 027b7b1623db | ba8b7e85ffea |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | tied | better | graph_prior_only | 40 | 8f87f425b630 | 2df5a9f45480 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | tied | better | graph_prior_only | 40 | 0b2c3595a8db | 6f16dff1b1ad |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | tied | better | graph_prior_only | 40 | 29fa972314df | ddab66e349e7 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | tied | better | graph_prior_only | 40 | 1b2ccc983296 | 9edaa3b10a9f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | tied | better | graph_prior_only | 40 | 83193ede922f | ebf44f1e0edf |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | worse | better | graph_prior_only | 60 | 3f83fc3271f3 | 1aa3f3c9b04c |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | tied | better | graph_prior_only | 60 | 25879dc6c7d4 | fe7f8b01f8fd |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | tied | better | graph_prior_only | 40 | 3ce3601e9ab1 | 38bfb0530318 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | tied | better | graph_prior_only | 40 | 7dd6446454ce | 015ed4563792 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | tied | better | graph_prior_only | 40 | 6bb0e3e81740 | 53550cd34dab |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | tied | better | graph_prior_only | 40 | edc4ef790493 | 9abc579bfb91 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | tied | better | graph_prior_only | 40 | 8f896cb74642 | b292fe396e6c |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | tied | better | graph_prior_only | 40 | f3d974e5afe6 | a8a674998b1f |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | tied | better | graph_prior_only | 40 | 7f28d0fd9eae | 4bbe27cd69f8 |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | tied | better | graph_prior_only | 40 | 3ffff4226634 | 00e97ec051de |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | worse | better | graph_prior_only | 60 | 1cd7085384e6 | f827fcaa99d6 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | tied | better | graph_prior_only | 40 | 82615d85cc1f | 550e271f5987 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | tied | better | graph_prior_only | 40 | 1a6c0e5779cc | b50a4d189166 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | worse | better | graph_prior_only | 60 | 9ff4706ea1b5 | b1b81c3fc0ab |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | tied | better | graph_prior_only | 40 | d6141bbd4a7e | e746be95d89d |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | tied | better | graph_prior_only | 40 | 220762ed2ecc | ec4df36d70a1 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | tied | better | graph_prior_only | 40 | 2c50b6bc4ca4 | f29d6bedc83d |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | tied | better | graph_prior_only | 40 | 32dfea621136 | 1f589eae80e3 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | tied | better | graph_prior_only | 40 | f2fd1a130619 | ac1516b8f95e |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | tied | better | graph_prior_only | 40 | 99af1a767a32 | b36c1b669008 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | tied | better | graph_prior_only | 40 | 1aa0c94cbc7a | d7cef5388be8 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | tied | better | graph_prior_only | 40 | 3ed855ae861e | 5ea34387ebe2 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | tied | better | graph_prior_only | 40 | 2cae4b38a10d | 89270136b785 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | tied | better | graph_prior_only | 40 | 530fc2867899 | efd4590009fb |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | tied | better | graph_prior_only | 40 | 4965293afb74 | b8385802c52b |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | tied | better | graph_prior_only | 40 | 043e56cb90ab | 80e7bfac8254 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | tied | better | graph_prior_only | 40 | 3e8109479b1d | 2b699f0e7ce7 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | tied | better | graph_prior_only | 40 | 6504ffd603e5 | 3cece2ba4bda |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | tied | better | graph_prior_only | 40 | 4a0fc15ec42c | 7280346b6f8e |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | tied | better | graph_prior_only | 40 | f240019be541 | 513a949bf00a |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | tied | better | graph_prior_only | 40 | 7d091dcbdbe8 | 0294e8db598d |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | tied | better | graph_prior_only | 40 | e10ab3c346ca | 1de544fcf08c |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | tied | better | graph_prior_only | 40 | dc0ad6ef426a | 47b4ba8f8b31 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | tied | better | graph_prior_only | 40 | af7adeb39e3a | e7bc1424bf58 |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | better | better | learned_route | 60 | a3ebc2601cd7 | 9e05a36eada2 |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | tied | better | graph_prior_only | 40 | d930045df8e5 | d1f174ebcfc2 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | tied | better | graph_prior_only | 40 | b74cac962543 | fa694a82dd6a |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | tied | better | vector_only | 70 | e74e33f8f725 | ef911039a7af |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | tied | better | graph_prior_only | 40 | 9734bbfda49d | 275e68e0b068 |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | worse | better | graph_prior_only | 60 | c333c7105123 | b5b865e0436c |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | tied | better | graph_prior_only | 40 | 350b2fa418da | e148bd8a2ba5 |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | tied | better | graph_prior_only | 40 | 99ef62b61e49 | df9a5f548512 |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | tied | better | graph_prior_only | 40 | d90a08585187 | 613941103123 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | tied | better | graph_prior_only | 40 | f1f1ae82d61e | ad540385fe49 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | tied | better | graph_prior_only | 40 | 25cd4ccd7fe3 | 01d670c3f859 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | tied | better | graph_prior_only | 40 | 73870c83f304 | 15664928dd7c |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | tied | better | graph_prior_only | 40 | c0289b336063 | fb3908d00ebb |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | tied | better | graph_prior_only | 40 | 552fc6a5d443 | 62f4a765857e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | tied | better | graph_prior_only | 40 | 5667277a8fd0 | 7428bdde1e7b |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | tied | better | graph_prior_only | 40 | fdb3f9845940 | 251023a835b4 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | tied | better | graph_prior_only | 60 | b4cdfbbc8557 | c7226daa0a4e |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | tied | better | graph_prior_only | 60 | 1c67492785ed | b5e18dc70f4c |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | tied | better | graph_prior_only | 40 | ac5cdb5b69ab | b6d21da85ba5 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | tied | better | graph_prior_only | 40 | 24a651bd6437 | e51f035b7ee7 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | tied | better | graph_prior_only | 40 | 4f4af7365e97 | ca02fc8ee0e3 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | tied | better | graph_prior_only | 40 | a6356cd81b0d | 960261aeefeb |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | tied | better | graph_prior_only | 40 | 4a847f4812af | c43743e4fbad |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | 7ccc960c1014 | fb35bd5499c3 |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | tied | better | graph_prior_only | 40 | 81ea9559bcd2 | 5f2458bff290 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | tied | better | graph_prior_only | 40 | 2158d8bc853a | 5d369844f156 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | tied | better | graph_prior_only | 40 | 42682ceba17d | 0efeeb994182 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | tied | better | graph_prior_only | 40 | c37648d4239b | 528f17d131a6 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | tied | better | graph_prior_only | 40 | d0b48cbe57d2 | d723d9cbabd0 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | tied | better | graph_prior_only | 40 | 4ce6e4535f6f | cba2f2e82f90 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | tied | better | graph_prior_only | 40 | a68ba20b0481 | a667bca9ae04 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | tied | better | graph_prior_only | 40 | 1eba7e0f2259 | 2b46f7828d61 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | tied | better | graph_prior_only | 40 | 07fb224e38b5 | 49eeeef5b671 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | tied | better | graph_prior_only | 40 | d67af96bce63 | e1bc5a0fbb1c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | tied | better | graph_prior_only | 40 | 1dcd0519d403 | 052273b81dd1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | tied | better | graph_prior_only | 40 | b4a07630a699 | bec7aa254a5f |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | tied | better | graph_prior_only | 40 | 233f80d8aa2f | 62c7e91dbf78 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | tied | better | graph_prior_only | 40 | 8cf5a38b1af7 | ace819cf7992 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | tied | better | graph_prior_only | 40 | 9e146d410b93 | 19903bca50b6 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | tied | better | graph_prior_only | 40 | d3dd64ede965 | d460d531e1d2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | tied | better | graph_prior_only | 40 | 74b2c61a3bda | f81138bf9b18 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | tied | better | graph_prior_only | 40 | 64d68bfbd5c7 | 7d39b5422512 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | tied | better | graph_prior_only | 40 | 1cf5d9a011ee | c8962bc5c099 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | tied | better | graph_prior_only | 60 | 3a1008e40e71 | 40da79b54ccf |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | tied | better | graph_prior_only | 40 | 92c96d3b2579 | 0f1f2c39d609 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | tied | better | graph_prior_only | 40 | 03cb9e1faa53 | 83ca91d92fcb |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | tied | better | graph_prior_only | 40 | 3feecb5a3785 | f70e8ec1997d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | tied | better | graph_prior_only | 40 | 48f997c984fb | d95e0b4e131b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | tied | better | graph_prior_only | 40 | f9c0751b0531 | c22f46ae2cd9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | tied | better | graph_prior_only | 40 | 3a312db11c5b | d04c006f8222 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | tied | better | graph_prior_only | 40 | e4819fc29284 | ea442e61a1d9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | tied | better | graph_prior_only | 40 | 9d79414ec5ac | 4f2c9b814b41 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | tied | better | graph_prior_only | 40 | 24f8867cc810 | 2f4a1efe5fcc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | tied | better | graph_prior_only | 40 | abf887a5acfe | 7d29d4347627 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | tied | better | graph_prior_only | 40 | 3f5508622b71 | b71145886588 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | tied | better | graph_prior_only | 40 | c83a1ee71b82 | cdbf9b07da91 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | tied | better | graph_prior_only | 80 | ae799d6db1ab | 38042e1cdafe |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | tied | better | graph_prior_only | 80 | a45001dba7ea | 3bb2d19dec5c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | tied | better | graph_prior_only | 40 | 622d47b59995 | ac83379138f7 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | tied | better | graph_prior_only | 80 | 6eeaa8bbc8b7 | bc73089bed9d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | tied | better | graph_prior_only | 40 | 457ba7981448 | ef7c435064b6 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | tied | better | graph_prior_only | 40 | f9903467ba63 | bcec87a2d127 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | tied | better | graph_prior_only | 40 | c5dd04d05cde | 8673bcd7bd77 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | tied | better | graph_prior_only | 40 | 9cd00d69e2bb | d9977552ee6b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | tied | better | graph_prior_only | 40 | 4f617066107b | d42546299763 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | tied | better | graph_prior_only | 40 | 7d02e1be688d | ba563d7516e5 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | tied | better | graph_prior_only | 40 | e51d02616462 | 09b32b7dbd4b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | tied | better | graph_prior_only | 40 | 5b1bdb0e2bf9 | aae39f7756a9 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | tied | better | graph_prior_only | 40 | 10504644ba63 | b11c6d446f99 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | tied | better | graph_prior_only | 40 | 04b55cc466a3 | 706356a36ab6 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | tied | better | graph_prior_only | 40 | 1393d86e45d6 | e878f64f1426 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | worse | better | graph_prior_only | 60 | 3c25ba359895 | 5f60ce15b942 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | tied | better | graph_prior_only | 40 | f48c30c639da | a9b6b29f6d68 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | worse | better | graph_prior_only | 60 | 4d52d86159fc | 9775ec92520e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | tied | better | graph_prior_only | 40 | 22ab6a8396d1 | c613572fffd2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | tied | better | graph_prior_only | 40 | 0503ff59e55d | 0a84e26e0b4c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | worse | better | graph_prior_only | 70 | 468f5e45121b | 977a626f9caf |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | tied | better | graph_prior_only | 40 | 99b615ab619c | efbad6f99af1 |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | tied | better | graph_prior_only | 40 | 4b45099c5fdb | d9eaedf8a0c3 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | tied | better | graph_prior_only | 40 | 90aebd4ca9fb | 35aaefbb07a7 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | tied | better | graph_prior_only | 40 | 9198b46a2c71 | 2b5a19fc7753 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | tied | better | graph_prior_only | 60 | 23de562adf6b | 3168fa1d4161 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | tied | better | graph_prior_only | 40 | 71de04aa68ee | 7c0a40bd9b8d |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | tied | better | graph_prior_only | 40 | 653ffaebf27b | 453b62ea089b |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | tied | better | vector_only | 60 | bee1055f2157 | d4e7cd58dfca |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | tied | better | graph_prior_only | 40 | c098f9a2f18c | 540833aec03f |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | worse | better | graph_prior_only | 70 | b917fb89ca9e | b6a0852c84ad |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | tied | better | graph_prior_only | 40 | 81b0be268415 | ab8edf4e6649 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | tied | better | graph_prior_only | 40 | 42020555ea34 | 80aa32618da7 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | tied | better | graph_prior_only | 40 | 5393c0c22d1f | b4b56846d9d4 |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | tied | better | graph_prior_only | 40 | 3aa24f7342d1 | 974bb4ede82a |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | tied | better | graph_prior_only | 40 | c9df5626a775 | 5052a766acd9 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | tied | better | graph_prior_only | 40 | e0d71bfa3306 | 114e46d24c42 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | tied | better | graph_prior_only | 40 | ca96faf84b51 | 5776296da341 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | tied | better | graph_prior_only | 40 | be54365f34b5 | d98bb2d851ef |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | tied | better | graph_prior_only | 40 | ac8fabbaaf68 | 70cc5f8dc99a |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | tied | better | graph_prior_only | 40 | 51d2415c88da | 3064f14fdd8c |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | tied | better | graph_prior_only | 40 | ab7f037f87cc | 45b22dee6d33 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | worse | better | graph_prior_only | 70 | c24b5a344c90 | 24b5cd88d9bf |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | tied | better | graph_prior_only | 40 | 60b401b6310c | a349c9b37fc9 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | tied | better | graph_prior_only | 40 | 7f68601867fa | f7589ead63c1 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | tied | better | graph_prior_only | 40 | 43a4fa166dba | 002f2037de42 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | tied | better | graph_prior_only | 40 | 3f0546a09a23 | d7eef573ed95 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | tied | better | graph_prior_only | 40 | 11b7a6f95ee0 | 29a6fcc1664d |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | tied | better | graph_prior_only | 40 | 2adc9e5c7748 | 8783fd743d09 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | tied | better | graph_prior_only | 40 | ae4ca8d2097d | e0941c429837 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | tied | better | graph_prior_only | 40 | f8efb4e158cc | 086dc1635420 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | tied | better | graph_prior_only | 40 | fe10589bb5e8 | 412d6123d267 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | tied | better | graph_prior_only | 40 | 5ea4e697c404 | a31f83372a20 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | tied | better | graph_prior_only | 40 | f6453c0b4fff | 6bf89e9f9e55 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | tied | better | graph_prior_only | 40 | 6c62ad607730 | 4a84fed6db70 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | tied | better | graph_prior_only | 40 | 246ee94676f9 | c335326e441f |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | tied | better | graph_prior_only | 40 | 890f0384472e | 4b227dc6c626 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | tied | better | graph_prior_only | 40 | e93d2c756611 | da1ffab2f4c4 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | tied | better | graph_prior_only | 40 | fc74ec3b3753 | b3dcf56df8af |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | tied | better | graph_prior_only | 40 | 9974545e8dee | 279e5705bc4c |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | tied | better | graph_prior_only | 40 | d84befc19540 | 0bb0c37dea0c |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | tied | better | graph_prior_only | 40 | 79aa239a8b66 | 42865e9bcd50 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | tied | better | graph_prior_only | 40 | b464d6265e54 | a2b47cf6f3c6 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | tied | better | graph_prior_only | 40 | d8d226b1582b | ebf21a99bac7 |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | tied | better | graph_prior_only | 40 | fe5cf29cf052 | 0b4be03bfab2 |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | tied | better | graph_prior_only | 40 | 75a4b5db3fc6 | 5e23b7331c7e |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-0776632a8e1027d76c1e2c79d6eb16efe1c19f49550f0730c06213259bfdcd3b |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-20b29791506fe72f146ea3338393fc30bb46be5bc14f8f5500ce44f09b8ff20b |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-ca815ec2cac155df12ec0f127785b93ef9ae5d1592c9eff453f95742ff22ac10 |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-85f48eebf10143ec632896402935dd3d0a73479ed950ba08f482dd647b43d822 |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ff9a5d7fa04a6ea40a5b485e20bfab28357447ca80fcda5e750a35c8fdaa5d83 |
| worked-traces | worked-traces.md | none | sha256-b7525021670bccc51081246d8875b9d38e5bf64a8298e87ca1e758c819b8ca51 |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-0a67cc863a6b2ca4c5ae273e9a2863687a8fefbc31da452b0ad102c542d26c40 |
