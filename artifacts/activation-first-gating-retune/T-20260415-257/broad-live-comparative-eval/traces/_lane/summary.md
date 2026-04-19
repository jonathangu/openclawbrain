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
- learned_route vs graph_prior_only (traces): 8 better, 395 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 403/403 (1)
- learned_route vs graph_prior_only (turns): 8 better, 395 tied, 0 worse
- regressions vs graph_prior_only: 0/403 (0)
- regressions vs no_brain floor: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 63/832 required-context phrases vs graph_prior_only 54/832
- correction absorption: correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here
- activation precision: explicit learned-route activation precision is 8/403 across 403 observed candidate turns
- activation precision proxy: selection-divergence proxy activation precision is 8/403 against graph_prior_only
- success-adjusted economics: learned_route used 395 estimated prompt tokens, 0.000494 estimated prompt USD, and 10 ms serve-path latency per incremental win vs graph_prior_only 236, 0.000295, and 10
- fail-open: observed 0/403 degraded learned_route turns in this replay lane

## Diagnostic Tie-Break Counts
| mode | diagnostic top-rank | shared top score traces |
| --- | ---: | ---: |
| no_brain | 0 | 0 |
| vector_only | 0 | 401 |
| graph_prior_only | 395 | 395 |
| learned_route | 8 | 403 |

## Trace Hashes
| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |
| --- | --- | --- | --- | ---: | --- | --- |
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | tied | better | graph_prior_only | 60 | 803bac354e7e | 3471adffaf4c |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | tied | better | graph_prior_only | 40 | a1ab4c941721 | a745b1a77d93 |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | tied | better | graph_prior_only | 70 | 079c673b5f9f | 242549a36165 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | better | better | learned_route | 80 | b3ed96e7ece7 | 5321d0771cdb |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | tied | better | graph_prior_only | 40 | 0a2b23c3d4a9 | 73c1ca2747de |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | tied | better | graph_prior_only | 60 | 62af66b5e3ea | d5a17c0f2703 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | tied | better | graph_prior_only | 60 | 52329e665a43 | 6cd2b94881c5 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | tied | better | graph_prior_only | 60 | 9b940ac35f5e | d5fe294638b9 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | tied | better | graph_prior_only | 60 | 2d61e1b986b8 | c2f68ad915e0 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | tied | better | graph_prior_only | 40 | 3a29e3759908 | cb1f7701b628 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | tied | better | graph_prior_only | 60 | a064ace52450 | 6ee0e4b1e750 |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | tied | better | graph_prior_only | 40 | b0a7cc07a414 | 15bc80d90fcf |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | tied | better | graph_prior_only | 40 | 775994e4fa40 | c1fe4f606c99 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | tied | better | graph_prior_only | 40 | 3324def8a9f9 | eaea4e40b3bf |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | tied | better | graph_prior_only | 40 | 647c44a3f15d | b2bfdf98fe4d |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | tied | better | graph_prior_only | 40 | 412a350761dc | e9f25c9f7d54 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | tied | better | graph_prior_only | 60 | 3cf1df6f34a9 | ed5b0bd2d5db |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | tied | better | graph_prior_only | 40 | e98e79ef3d6e | 1af4b5e51d52 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | tied | better | graph_prior_only | 40 | 7d94460b126f | 520397b99f46 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | tied | better | graph_prior_only | 40 | 1693902b1cc6 | 14727f8ebe70 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | tied | better | graph_prior_only | 40 | 9be3b0aaf9c8 | dd52c7b8756c |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | tied | better | graph_prior_only | 100 | 394e91df1b56 | 98cdf0ff93e2 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | tied | better | graph_prior_only | 40 | b088cb9ebba5 | 45f385cbbba3 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | tied | better | graph_prior_only | 40 | 27e8fd6e64f4 | 8da328b6a5ef |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | tied | better | graph_prior_only | 100 | af2d1b062db8 | c19db0d0e3d8 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | tied | better | graph_prior_only | 40 | 01a28ea088f0 | e1d106606cee |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | tied | better | graph_prior_only | 40 | 69826a259772 | 824be16ff05d |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | tied | better | graph_prior_only | 100 | 8a7a2bb5d4da | 00cec6c59050 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | tied | better | graph_prior_only | 40 | f981ae380869 | 8c6219f2e16b |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | tied | better | graph_prior_only | 40 | 8cbee3df93b0 | d9cbafe3bafc |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | tied | better | graph_prior_only | 40 | dd856b59f3a2 | 492a7f7f612a |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | tied | better | graph_prior_only | 40 | 19e915e1419f | 98127d0c1992 |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | tied | better | graph_prior_only | 60 | 1f70fa8e7cb4 | 5e94e64a4117 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | tied | better | graph_prior_only | 40 | 14cfdfcf3aca | d567026d4472 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | tied | better | graph_prior_only | 40 | ace730faea6e | e1cb27738724 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | tied | better | graph_prior_only | 40 | c71b087aa2d9 | f3ca94287f6b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | tied | better | graph_prior_only | 40 | 2a0ab062d186 | 1545f5ad31ec |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | tied | better | graph_prior_only | 40 | e299c13ff4e0 | 6cda9367d323 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | tied | better | graph_prior_only | 40 | 4f2a8ac77aab | 090d8c712616 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | tied | better | graph_prior_only | 40 | e5b292f692f6 | 7db955fe0c2c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | tied | better | graph_prior_only | 40 | d15408a1be9f | 0feeb7de0056 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | tied | better | graph_prior_only | 40 | 3cc4629e2deb | dc4ef25689aa |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | tied | better | graph_prior_only | 40 | 8e057c83a1dc | 047db00fe951 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | tied | better | graph_prior_only | 40 | f58a15910288 | fa377f04f984 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | tied | better | graph_prior_only | 40 | 3dd85795110f | c98cbac39644 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | tied | better | graph_prior_only | 40 | 318d53c011a4 | 3dc4ed6031bd |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | tied | better | graph_prior_only | 40 | 5ef373f8c69d | fdbebb10bf36 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | tied | better | graph_prior_only | 40 | 6fa78fe3cb6a | 48790daab17d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | tied | better | graph_prior_only | 40 | 014b7a38aa86 | fcfe0b98657b |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | tied | better | graph_prior_only | 40 | 77d9d8e2b26c | fc1172ba93a8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | tied | better | graph_prior_only | 60 | 264ab5e8fa4d | 2bdcc3db7dcc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | tied | better | graph_prior_only | 40 | 83a1e2f7f88c | 334285f301f0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | tied | better | graph_prior_only | 40 | 35c155280737 | 5d188626b1e8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | tied | better | graph_prior_only | 60 | f761457b7a52 | a6bcd43aa9b0 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | tied | better | graph_prior_only | 40 | 43473ede2b0c | c8a60b2f774f |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | tied | better | graph_prior_only | 40 | 029f2ce3ddf4 | f5a6b96ac089 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | tied | better | graph_prior_only | 40 | 5163aa4ebe03 | fa6659678232 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | tied | better | graph_prior_only | 40 | 9978f10e4089 | 0bbcb091c40d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | tied | better | graph_prior_only | 40 | c3376547faec | c22509062696 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | tied | better | graph_prior_only | 40 | 4a53d15db398 | 81ef3b2908f2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | tied | better | graph_prior_only | 40 | 27f823a5d784 | ae1b3bba69b1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | tied | better | graph_prior_only | 40 | 624f122899aa | 0c6f39eda9d2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | tied | better | graph_prior_only | 40 | fbaed24c1342 | e719c328f852 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | tied | better | graph_prior_only | 40 | 1a1b695ddfa2 | 05095924e0d1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | tied | better | graph_prior_only | 40 | 9a29cb519cfe | bd1aa67a14fb |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | tied | better | graph_prior_only | 40 | 4bf5563c267b | 9b60ad0334da |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | tied | better | graph_prior_only | 40 | 21aab2be51b2 | 967b635c9dc4 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | tied | better | graph_prior_only | 40 | dc6fade7e705 | 0486e6353f12 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | tied | better | graph_prior_only | 60 | 4760fc810e06 | 0131fe184aaa |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | tied | better | graph_prior_only | 40 | fad4d6d9221c | cad5a04e9e7f |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | tied | better | graph_prior_only | 40 | 20aa2e8563b1 | bacadc7181fc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | tied | better | graph_prior_only | 40 | 21a3f39117ea | 53d819b9375a |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | tied | better | graph_prior_only | 40 | 6b4a19b823c6 | 92235445d0d8 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | tied | better | graph_prior_only | 40 | ee181856a2b7 | c05ee52f277f |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | tied | better | graph_prior_only | 40 | b439db87f923 | 4871ffbe318c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | tied | better | graph_prior_only | 40 | 3890dea55662 | 51b14c7df098 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | tied | better | graph_prior_only | 40 | 3039b56f6516 | 66836cfca98d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | tied | better | graph_prior_only | 60 | 05dc50592f72 | 169f11bdf747 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | tied | better | graph_prior_only | 40 | 2d8c26cdb6d7 | 556287d12b21 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | tied | better | graph_prior_only | 40 | 2b9b68f5782f | 44077c424629 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | tied | better | graph_prior_only | 40 | a2c45b56c2e3 | 9a38ee5ec1b1 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | tied | better | graph_prior_only | 40 | 6223077caa27 | 0a78064305e2 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | tied | better | graph_prior_only | 40 | bf802c2cb2e5 | b38459c0913c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | tied | better | graph_prior_only | 40 | f9271b6b1771 | b3e4c92b6f06 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | tied | better | graph_prior_only | 40 | 55b5cfcfa3d8 | 30671e2c9d9c |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | tied | better | graph_prior_only | 40 | f442b7501697 | 5ba828aaa283 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | tied | better | graph_prior_only | 40 | 860021b2cebd | 6e6a9bfb6cdc |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | tied | better | graph_prior_only | 40 | f8618aeb0532 | b66e90bcdc75 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | tied | better | graph_prior_only | 40 | f9cdede5b807 | 77ac7d0bdf28 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | tied | better | graph_prior_only | 40 | 2a518be35f72 | 56fd5f6dd428 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | tied | better | graph_prior_only | 40 | 5fec7511593c | 2239aa0819cf |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | tied | better | graph_prior_only | 40 | 84d34ce84b34 | 57fd2fd5b7a7 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | tied | better | graph_prior_only | 40 | 95e6bf92344d | 391db499a839 |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | tied | better | graph_prior_only | 40 | 1ac3dc77ebc5 | f85964dab02d |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | tied | better | graph_prior_only | 70 | c8012b171b0b | 129eee972321 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | tied | better | graph_prior_only | 40 | 98193542f14c | 586262eb7cad |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | tied | better | graph_prior_only | 40 | 188dbda27d67 | 036ea37ca664 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | tied | better | graph_prior_only | 40 | abb1ea78767b | 6645fc40a4c0 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | tied | better | graph_prior_only | 40 | de5f7de8d781 | 41a4a4c810de |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | tied | better | graph_prior_only | 40 | fdeb0a6c6a9d | 7baab078c2c3 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | tied | better | graph_prior_only | 40 | df4e3ab6ade6 | 2e0c83aa8825 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | tied | better | graph_prior_only | 40 | 05cbfc3b0649 | c5580121775b |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | tied | better | graph_prior_only | 40 | 9faf3b0e7b10 | a19a8d94b09b |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | tied | better | graph_prior_only | 40 | e183bc718be4 | ffce6affbb6f |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | tied | better | graph_prior_only | 40 | 3615838bc291 | 7002e7bc5163 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | tied | better | graph_prior_only | 40 | e2ff2bcd0ce9 | 0970a78a2c47 |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | tied | better | graph_prior_only | 40 | 229bb2585504 | b94033cddacc |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | tied | better | graph_prior_only | 40 | b65de8a16c15 | c7be25bbf930 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | tied | better | graph_prior_only | 40 | cc3f3ab98246 | 7737ba490141 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | tied | better | graph_prior_only | 40 | c427b697a559 | 5aa5ba3083d0 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | tied | better | graph_prior_only | 60 | 5d9e8c3d28be | edb7744c0f9b |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | tied | better | graph_prior_only | 40 | 754a6a68ccba | 8b12b6eb46e9 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | tied | better | graph_prior_only | 40 | 2a5247aaf5f4 | 1e25862836b2 |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | tied | better | graph_prior_only | 40 | 573136fe2689 | 7848d1008d8e |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | tied | better | graph_prior_only | 100 | c2cc00d2a352 | 46eb6a3baaf6 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | tied | better | graph_prior_only | 40 | 549fc664d431 | 379a806b133b |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | tied | better | graph_prior_only | 40 | 6873378caaa0 | dd4dd8a90cda |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | tied | better | graph_prior_only | 40 | 1b74bdaccd00 | f063ba89e56f |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | tied | better | graph_prior_only | 70 | 326fb2a5faca | 62736b6c49e4 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | tied | better | graph_prior_only | 40 | b1b2a4962d7e | cbec8ac58cfd |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | tied | better | graph_prior_only | 40 | 146fb39bdcc9 | 1c97a3bd5fbd |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | tied | better | graph_prior_only | 40 | 1505427d0c55 | 10856e4224fd |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | tied | better | graph_prior_only | 40 | 6c01b851c4c3 | 8cd94db0e06a |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | tied | better | graph_prior_only | 40 | 7fdb9bd608f5 | 400b0c723835 |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | tied | better | graph_prior_only | 40 | 2be5417cf8c5 | 1349e3057bf2 |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | tied | better | graph_prior_only | 40 | 3af3f9188548 | d9d029813153 |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | better | better | learned_route | 100 | 9938661101d6 | cc7bce9fc4f6 |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | tied | better | graph_prior_only | 40 | 3d755350366e | 3c9323571781 |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | tied | better | graph_prior_only | 40 | d1f58811f2d1 | 601e0831c419 |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | tied | better | graph_prior_only | 40 | 0e7f0019a5ab | 52f8be4d626f |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | tied | better | graph_prior_only | 80 | bce48fd5e3ac | d38c0c85951f |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | tied | better | graph_prior_only | 40 | 9baf2df17280 | 8e83b14cae72 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | tied | better | graph_prior_only | 40 | 20a8deb4dfe4 | 3ffe30b2f0cc |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | better | better | learned_route | 60 | 448a53032769 | 22b3ccf1c477 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | tied | better | graph_prior_only | 40 | f32cf2298b0b | 0767e58794ee |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | tied | better | graph_prior_only | 40 | 631db590ba34 | 46f6c565455b |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | tied | better | graph_prior_only | 40 | 62fb4710dc11 | 16b81f928258 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | tied | better | graph_prior_only | 40 | 881a492dec80 | 23428e727858 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | tied | better | graph_prior_only | 40 | 379d5f18ff8a | 6dd7a2a02230 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | tied | better | graph_prior_only | 40 | 860c9b254aaf | 9b62fc64c6ea |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | tied | better | graph_prior_only | 40 | a8f2956e6207 | 8414971f007b |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | tied | better | graph_prior_only | 40 | ba8cea8cefa3 | d877e2f479f3 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | tied | better | graph_prior_only | 40 | de6e38d24965 | 1fe6fae980d5 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | tied | better | graph_prior_only | 40 | 7ca4baedd626 | d95def292127 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | tied | better | graph_prior_only | 40 | 18e279867752 | 9e18d3f9c8a6 |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | tied | better | graph_prior_only | 40 | b40d74bbc951 | febd388fde8a |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | tied | better | graph_prior_only | 40 | b2180ea31efb | 74b22e9a07cb |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | tied | better | graph_prior_only | 40 | f3508c06188a | 627c585a5f8a |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | tied | better | graph_prior_only | 40 | 99f01787f6fa | a40e08e630cc |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | tied | better | graph_prior_only | 40 | 2e6fbf56ad6b | 9a2090834ad2 |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | tied | better | graph_prior_only | 80 | 2fb08c4c7874 | 492d374b42d9 |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | tied | better | graph_prior_only | 60 | e3c6fbb52635 | 541c5537b667 |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | tied | better | graph_prior_only | 40 | 8216a85a23bf | 05cb0d99c32e |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | tied | better | graph_prior_only | 40 | 72eeab4fc992 | 5851f2c04b5a |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | tied | better | graph_prior_only | 40 | 048fa4d8a6e0 | 4b43c242eddf |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | tied | better | graph_prior_only | 40 | e5dd729cd900 | b4677d7aac3f |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | tied | better | graph_prior_only | 40 | f72ea1192ca3 | 7d31b9d029c3 |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | tied | better | graph_prior_only | 40 | eb4f3bc8d8c7 | da8fd3bcdd9a |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | tied | better | graph_prior_only | 60 | dbf4ead04041 | bb681ea566ce |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | tied | better | graph_prior_only | 40 | ebe777cc720e | c54394de7c90 |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | tied | better | graph_prior_only | 100 | a9dd31523ee3 | b3a7643f6dfe |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | better | better | learned_route | 70 | 6f22d316cbdb | 8770de740d04 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | tied | better | graph_prior_only | 40 | fb5552dda9ab | ab246c9d70d6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | tied | better | graph_prior_only | 40 | 0e89ae5a8b2a | bf08c0f511db |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | tied | better | graph_prior_only | 40 | d549bb42e869 | 86e66d3e9634 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | tied | better | graph_prior_only | 40 | f7bd1e31f3aa | 7a23d55060fa |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | tied | better | graph_prior_only | 40 | 609b15d03bd2 | cefca9ca9a1c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | tied | better | graph_prior_only | 40 | 44d454b7ce2a | 9ab5e0aac5da |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | tied | better | graph_prior_only | 40 | 5b4093839eb4 | dd5a4231e51c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | tied | better | graph_prior_only | 40 | b186a958bf09 | 2a853f0de058 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | tied | better | graph_prior_only | 40 | ad8ceacd9266 | b8290fa0de39 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | tied | better | graph_prior_only | 40 | 5e250412f4eb | 16a539061e0c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | tied | better | graph_prior_only | 40 | a58501fd2f54 | 1378e805e16c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | better | better | learned_route | 100 | bc1ae43a24ef | 9b12e9e169db |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | tied | better | graph_prior_only | 100 | a7261a2a9494 | dad92fdc522f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | tied | better | graph_prior_only | 40 | e02319b69b2b | ed41b50c86df |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | tied | better | graph_prior_only | 40 | 4ad1ed5f3026 | a3ba083a0f89 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | tied | better | graph_prior_only | 40 | ac06c847087a | 7ee4b714b1ad |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | tied | better | graph_prior_only | 40 | 67510c926563 | ea006f9add83 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | tied | better | graph_prior_only | 40 | fbd919949534 | 45a8c84dbba5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | tied | better | graph_prior_only | 40 | d63598f1458b | afd4c0c95192 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | tied | better | graph_prior_only | 40 | dcf9756363f1 | 036563d0ea47 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | tied | better | graph_prior_only | 40 | 7d8bd172af2a | eaad48d111d1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | tied | better | graph_prior_only | 40 | 27d7c4956567 | 07685ca5ced4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | tied | better | graph_prior_only | 40 | 853ba8a8415a | bdafbfb1f3df |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | tied | better | graph_prior_only | 40 | c0b2ac08bb1f | e047aa7ae9b9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | tied | better | graph_prior_only | 40 | 63e38c2ecbe0 | 31555b831150 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | tied | better | graph_prior_only | 40 | ea8988a21087 | 59b39897b24f |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | tied | better | graph_prior_only | 40 | 4053587c326f | c9e18e2068ec |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | tied | better | graph_prior_only | 40 | a83a1e9314a0 | 12ec35564965 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | tied | better | graph_prior_only | 100 | 5122bc93cd9f | 02900a2958e9 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | tied | better | graph_prior_only | 40 | fd24aeb7ff52 | 5c25ffeba817 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | tied | better | graph_prior_only | 40 | 1231ca5b529c | 6419d00a71d2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | tied | better | graph_prior_only | 40 | a2532141dbfc | c56646b007a1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | tied | better | graph_prior_only | 40 | 4cefdbe487de | e9a1595fc2ab |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | tied | better | graph_prior_only | 40 | 894b1a03f376 | 1f206da5d97c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | tied | better | graph_prior_only | 40 | 37cd75a2abaf | 82bbb412cfd0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | tied | better | graph_prior_only | 40 | 2d05a7ae173c | ee694b1967b8 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | tied | better | graph_prior_only | 40 | ce77bb1bd7b2 | f6fec12ef739 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | tied | better | graph_prior_only | 40 | fb8f148b16ea | 727ee0d5159d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | tied | better | graph_prior_only | 40 | 7dbb78285c78 | e71464a34c50 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | tied | better | graph_prior_only | 40 | 10dec07798ce | c6f36f1931f4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | better | better | learned_route | 100 | b1735144e48b | 2c93ea7fc788 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | tied | better | graph_prior_only | 40 | 30099c99d373 | 639b5aa1a326 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | tied | better | graph_prior_only | 40 | 1feed8db81a2 | c3f1c0abb646 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | tied | better | graph_prior_only | 40 | 95e848b0d86a | a564c0663d30 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | tied | better | graph_prior_only | 40 | a6f9a5fe54b1 | 4f9f752c080b |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | tied | better | graph_prior_only | 40 | 228840b84bd6 | 6ad69eb3f9bd |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | tied | better | graph_prior_only | 40 | 975d577a2687 | 480731907af0 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | tied | better | graph_prior_only | 40 | c4a0f9b1f7b3 | 94972ed210e1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | tied | better | graph_prior_only | 40 | 80c13c63d2d3 | d93c9ddf0c95 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | tied | better | graph_prior_only | 40 | 6dc792809af5 | 9c1daf5962b6 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | tied | better | graph_prior_only | 40 | add4165a428b | 51c3c31a35cb |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | tied | better | graph_prior_only | 40 | c4f4ba778e7f | 93305dd87def |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | tied | better | graph_prior_only | 40 | 6a50b63e6621 | 07fb32b7a48d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | tied | better | graph_prior_only | 40 | a6d001f9ea27 | 5a5527d6abe2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | tied | better | graph_prior_only | 40 | 2a9d5d3739c9 | 0c3a0207a2cd |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | tied | better | graph_prior_only | 40 | d1fb11621cca | 6890216e6283 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | tied | better | graph_prior_only | 40 | e599afb6e329 | 5166b36f1f2c |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | tied | better | graph_prior_only | 40 | 79cbe7ece0bc | 866a4f952ce1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | tied | better | graph_prior_only | 40 | 82a580aa50ac | 9114c292de16 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | tied | better | graph_prior_only | 40 | ed6d53cead41 | 22d10977243a |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | tied | better | graph_prior_only | 40 | 9a25dcc9b229 | 3efddf28b2a1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | tied | better | graph_prior_only | 40 | 1a8060062684 | 56d74a2844fe |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | tied | better | graph_prior_only | 40 | f856122a7334 | 83494e6cdb1e |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | tied | better | graph_prior_only | 40 | 0e14f6f7dfae | 143e3862a705 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | tied | better | graph_prior_only | 40 | 18d1aa215111 | 66d99cb09134 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | tied | better | graph_prior_only | 40 | 908ac44fa433 | 400e38a8ed39 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | tied | better | graph_prior_only | 40 | 66846596871b | 101cac1d5200 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | tied | better | graph_prior_only | 40 | 456b3fbfdabd | 141eff9ad1f2 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | tied | better | graph_prior_only | 40 | a7122eee81e9 | 7e508b74e89d |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | tied | better | graph_prior_only | 40 | 181efdb9532e | e99fff90da93 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | tied | better | graph_prior_only | 40 | 51c01c39b91d | 97bb932ccaa5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | tied | better | graph_prior_only | 40 | d6d827174c2f | 991a099e6003 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | tied | better | graph_prior_only | 40 | 43cb0821e843 | 718085fb6223 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | tied | better | graph_prior_only | 40 | d2ca470d184c | 4bfde2e04a81 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | tied | better | graph_prior_only | 40 | f29f210ea77e | ba8e938710b1 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | tied | better | graph_prior_only | 40 | 561777aed4e4 | 4e0ee79f80dd |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | tied | better | graph_prior_only | 40 | 2c7d367b88e1 | ddba2047fd79 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | tied | better | graph_prior_only | 40 | a89e8633cdc0 | e814551587a4 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | tied | better | graph_prior_only | 40 | 38e9eff36813 | a425e34f3ad5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | tied | better | graph_prior_only | 40 | 4bb789001730 | f3daeff21cc5 |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | tied | better | graph_prior_only | 40 | 4839e729fecb | 499806d2ecc2 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | tied | better | graph_prior_only | 40 | bb92ac0d4f10 | a241081288fd |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | tied | better | graph_prior_only | 40 | 45ff23b23e5d | ee13003e80d1 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | tied | better | graph_prior_only | 40 | df1b9f5c9718 | 3c05247e637d |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | tied | better | graph_prior_only | 40 | 483d4d9830e4 | fc92de29a602 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | tied | better | graph_prior_only | 60 | ab79ebbbe07e | fd0556bd5f58 |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | tied | better | graph_prior_only | 60 | 106983959421 | 3c60e3e4123e |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | tied | better | graph_prior_only | 40 | 2ec109c91af8 | 3c9cf15df57f |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | tied | better | graph_prior_only | 40 | 8f4dc4823172 | 055ef8ed0835 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | tied | better | graph_prior_only | 40 | 3b4950a7460b | 52def7564628 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | tied | better | graph_prior_only | 40 | 53e5746faa6f | 004041af13bf |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | tied | better | graph_prior_only | 40 | 7f6def1b44d3 | 99714a909e65 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | tied | better | graph_prior_only | 40 | 7074f333b7b4 | 05cf0e368681 |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | tied | better | graph_prior_only | 40 | 0d2b4f22c39a | 3f98d376af88 |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | tied | better | graph_prior_only | 40 | 3e888bbf88f6 | 5295f0d6b609 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | tied | better | graph_prior_only | 60 | 807f03f08f74 | dd119b9b57e8 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | tied | better | graph_prior_only | 40 | 9096f5b2e0f6 | be6b7552f8e3 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | tied | better | graph_prior_only | 40 | 742023a5eb9f | 203f6059c1c5 |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | tied | better | graph_prior_only | 60 | 038f480825b5 | 24597845535e |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | tied | better | graph_prior_only | 40 | 59e4bd501eeb | 7923e55b33ad |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | tied | better | graph_prior_only | 40 | 06db8b5b11b8 | 2cbe1ed87db3 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | tied | better | graph_prior_only | 40 | a08c583c21f1 | 7c9974efbd50 |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | tied | better | graph_prior_only | 40 | d2d979a50f29 | f79b2d5098a0 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | tied | better | graph_prior_only | 40 | e3a4da21d16f | b87af33e42be |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | tied | better | graph_prior_only | 40 | fe54af7d00c7 | e2ff22451b0d |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | tied | better | graph_prior_only | 40 | 75a96a286b5b | 8b8cd9fdba2f |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | tied | better | graph_prior_only | 40 | b73e9a949c60 | a0909223b640 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | tied | better | graph_prior_only | 40 | 5f3b8318e24c | bcdb5d57eef4 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | tied | better | graph_prior_only | 40 | 7eae589adc0f | d45e0e808256 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | tied | better | graph_prior_only | 40 | 08956d3a5a69 | 90bd4189be6b |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | tied | better | graph_prior_only | 40 | 2291ab48a31b | da895110e21e |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | tied | better | graph_prior_only | 40 | 1fdcc837078d | 1f39ec727ff9 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | tied | better | graph_prior_only | 40 | 960379cc9fb6 | 70fc691f26ac |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | tied | better | graph_prior_only | 40 | 7aeef0c2fb74 | c3c27b88b0c3 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | tied | better | graph_prior_only | 40 | fa5cdeb3715e | 995484c5db4e |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | tied | better | graph_prior_only | 40 | 5fa791661a2e | 8817aef4edf8 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | tied | better | graph_prior_only | 40 | 544c73333b5a | 3f665a16c156 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | tied | better | graph_prior_only | 40 | 6c8ceba4f009 | 77efb4cedc45 |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | tied | better | graph_prior_only | 40 | 662c6c3bc663 | 300930d801fe |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | better | better | learned_route | 60 | c4b0c1208c32 | 2917f27e7605 |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | tied | better | graph_prior_only | 40 | 0c54c50d198a | 97bdc46ad4ce |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | tied | better | graph_prior_only | 40 | 7cd9d0040095 | 326fa7d66a76 |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | tied | better | graph_prior_only | 70 | c7a145d599a1 | 4b46fcebe136 |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | tied | better | graph_prior_only | 40 | 1381eadc3692 | 9617c03dacb6 |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | tied | better | graph_prior_only | 60 | 983f17b827ae | 9efa4ed4d6ea |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | tied | better | graph_prior_only | 40 | 520cf2059f67 | dc46e9fd6369 |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | tied | better | graph_prior_only | 40 | 2efefc747655 | 681ffcc5623d |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | tied | better | graph_prior_only | 40 | 94877b882f73 | aff3a4ba6bb2 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | tied | better | graph_prior_only | 40 | 76e56af4ee9b | dac5e2d719df |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | tied | better | graph_prior_only | 40 | 0c8b1507426d | 8935d4b996af |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | tied | better | graph_prior_only | 40 | 9771bef4eff8 | abb08795eb2a |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | tied | better | graph_prior_only | 40 | 5356a25dd917 | 1c24d061080f |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | tied | better | graph_prior_only | 40 | 5699de113147 | b4acbb934f4c |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | tied | better | graph_prior_only | 40 | 986d157e079f | a9f6b28edfab |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | tied | better | graph_prior_only | 40 | 7ba7e547cc29 | be19cb55d0e2 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | tied | better | graph_prior_only | 60 | 93387397880d | cdb393d51448 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | tied | better | graph_prior_only | 60 | 38925b829de7 | 91c1b4b80156 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | tied | better | graph_prior_only | 40 | dde712d7de18 | fcb534c8d238 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | tied | better | graph_prior_only | 40 | 3352886415db | a7a6e2ee12ed |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | tied | better | graph_prior_only | 40 | bd0b4c64be77 | 4397c22fe5ac |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | tied | better | graph_prior_only | 40 | 66a9a35eda15 | e894b4359fa6 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | tied | better | graph_prior_only | 40 | 13a6d4dbac16 | 7cfb70041bd5 |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | tied | better | graph_prior_only | 40 | aff6dafbf9c4 | 5157cb755108 |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | tied | better | graph_prior_only | 40 | b58452a66d5e | a9cfd854f80e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | tied | better | graph_prior_only | 40 | 2c4579ab5b98 | 1b8fbce0d9aa |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | tied | better | graph_prior_only | 40 | fb2506e7608c | 562013d4a034 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | tied | better | graph_prior_only | 40 | 68bac9843505 | c4013a42622e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | tied | better | graph_prior_only | 40 | 3c45ee25b2e1 | 6d24cdbe2cd3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | tied | better | graph_prior_only | 40 | b733082941b3 | ec32b731295d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | tied | better | graph_prior_only | 40 | 19fff8c5d93c | cafac16fe06b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | tied | better | graph_prior_only | 40 | abfba73887ce | 4c6da2c62e24 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | tied | better | graph_prior_only | 40 | db59cf6f5d5b | c5a42b109d26 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | tied | better | graph_prior_only | 40 | 2a398f143d9a | cfd9b974ed35 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | tied | better | graph_prior_only | 40 | cf6e05a36a41 | c906445e3332 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | tied | better | graph_prior_only | 40 | 5cd852435075 | 1572bba8370b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | tied | better | graph_prior_only | 40 | 2e37917eaf04 | 632619dbdd94 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | tied | better | graph_prior_only | 40 | 8b6e2611ebc3 | 21bbc840ec97 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | tied | better | graph_prior_only | 40 | 73b34cf68f58 | cdb37f9e8d39 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | tied | better | graph_prior_only | 40 | 30010efc443d | e270f7c8d9fc |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | tied | better | graph_prior_only | 40 | 64354301bd72 | 13cf4f7dc584 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | tied | better | graph_prior_only | 40 | fdcb06349db4 | dac0f06aa306 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | tied | better | graph_prior_only | 40 | 688e8117ee76 | f3fcb76f0a7a |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | tied | better | graph_prior_only | 60 | 850d30534f8c | 04f23da07da2 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | tied | better | graph_prior_only | 40 | e9082655bdb0 | a55a8920572d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | tied | better | graph_prior_only | 40 | 37044cb81763 | 630410787fa7 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | tied | better | graph_prior_only | 40 | 22ebb848519d | b4075dfbcd55 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | tied | better | graph_prior_only | 40 | 1a80572fa1fa | 64e02c76ae6e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | tied | better | graph_prior_only | 40 | 59b37e2e2438 | 363713fdc744 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | tied | better | graph_prior_only | 40 | 2fd271769d69 | 9e6e319eb8eb |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | tied | better | graph_prior_only | 40 | 460c4802663b | 0033f042e03c |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | tied | better | graph_prior_only | 40 | b690df7592d3 | 81d8c3c68cd3 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | tied | better | graph_prior_only | 40 | fa83ae6956ed | d65d6575c561 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | tied | better | graph_prior_only | 40 | d2ac2a8fb475 | 83e0fe0d67f1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | tied | better | graph_prior_only | 40 | d5f13942ef43 | 3c665e9ecb27 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | tied | better | graph_prior_only | 40 | abff4595af21 | f2ac16d4d670 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | tied | better | graph_prior_only | 80 | 1f70f4b8e8fa | 1ab1f58c9e22 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | tied | better | graph_prior_only | 80 | 7218f54ad954 | 8b8b3048c431 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | tied | better | graph_prior_only | 40 | 07ce0a93c480 | 89e20abc38ae |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | tied | better | graph_prior_only | 80 | 211614674fb2 | 71eb300b2e61 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | tied | better | graph_prior_only | 40 | 00bfe80bdda9 | 40af921d11c6 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | tied | better | graph_prior_only | 40 | 93e66925c43f | 038fffcdc761 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | tied | better | graph_prior_only | 40 | 37c09211c96a | 272b01883857 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | tied | better | graph_prior_only | 40 | 5b5fdbd42d9c | d3fb76bed949 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | tied | better | graph_prior_only | 40 | 110c1a48adeb | 96cb49dfd74b |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | tied | better | graph_prior_only | 40 | ef3061cd39fe | 1b0ef3c7d1ab |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | tied | better | graph_prior_only | 40 | 05ebce2649e4 | ae175bf07a8e |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | tied | better | graph_prior_only | 40 | a7accf758bfb | 7c50b2bd5669 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | tied | better | graph_prior_only | 40 | 13e59636394e | 37dcfe4773f8 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | tied | better | graph_prior_only | 40 | 62c24e760f5a | 7c03143022b1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | tied | better | graph_prior_only | 40 | f83e6c23a042 | ef6e803dabb0 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | tied | better | graph_prior_only | 60 | d61e8a3c8f0b | 09ca8d6b3194 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | tied | better | graph_prior_only | 40 | 275e0dc9b0f8 | af9e9380e284 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | tied | better | graph_prior_only | 60 | 04b232933a36 | 8ad09962cc83 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | tied | better | graph_prior_only | 40 | ae7484c25548 | 7180419db22d |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | tied | better | graph_prior_only | 40 | 4db35d4abc8e | 14442397ecc1 |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | tied | better | graph_prior_only | 70 | 0a29006dffc6 | 618cf383abaa |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | tied | better | graph_prior_only | 40 | 1cf0f9cac858 | 9417f341b209 |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | tied | better | graph_prior_only | 40 | febda7527d18 | 3efb55057cec |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | tied | better | graph_prior_only | 40 | 5ef8c982564a | c42ffdc5b1a8 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | tied | better | graph_prior_only | 40 | 7f7373439a0b | 7ed9f9057810 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | tied | better | graph_prior_only | 60 | acd44cd41a23 | 135af4b6fb44 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | tied | better | graph_prior_only | 40 | aeeed3d04951 | 9edaf7668c74 |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | tied | better | graph_prior_only | 40 | b1df48e7e3b3 | c9c552955668 |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | better | better | learned_route | 60 | 0e938d173237 | 0beadbaf4897 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | tied | better | graph_prior_only | 40 | 5deb3c4d946c | 0e23b7e7a6e6 |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | tied | better | graph_prior_only | 70 | 2e66aa349a84 | cf148560a994 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | tied | better | graph_prior_only | 40 | 86de8be67fd7 | 0b47f1c8c077 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | tied | better | graph_prior_only | 40 | 85ae3de9e5c0 | b02aafcb1193 |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | tied | better | graph_prior_only | 40 | 62f2621bb0a1 | 6270ce9ea4da |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | tied | better | graph_prior_only | 40 | a410c51284e5 | 9d2b705f0d7a |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | tied | better | graph_prior_only | 40 | a27a02d28bee | b736f00d220b |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | tied | better | graph_prior_only | 40 | bd740a6bfb07 | 6c24a410d665 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | tied | better | graph_prior_only | 40 | c341f4783f05 | c59322571630 |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | tied | better | graph_prior_only | 40 | fd656b78d61a | 1952e16c3003 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | tied | better | graph_prior_only | 40 | cd0e59b2571f | 6171cc0c7d14 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | tied | better | graph_prior_only | 40 | bc0796da4b3a | 936902151405 |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | tied | better | graph_prior_only | 40 | 73ada8587b53 | f14a9f6e792b |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | tied | better | graph_prior_only | 70 | 47fdfc24d90a | 0db8d8b76ee8 |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | tied | better | graph_prior_only | 40 | eee03c6b1144 | aface03cec17 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | tied | better | graph_prior_only | 40 | 5a5f639ad7ba | 78f6d0758cc7 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | tied | better | graph_prior_only | 40 | 15146b30416a | 08d12ff71c12 |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | tied | better | graph_prior_only | 40 | decd6f1b19c0 | 823480aaa80d |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | tied | better | graph_prior_only | 40 | a9b3b8e06a8e | 53617e3983dd |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | tied | better | graph_prior_only | 40 | 1365619f2861 | ed906bbaf31e |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | tied | better | graph_prior_only | 40 | 69c6f046940c | c57dd87f502e |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | tied | better | graph_prior_only | 40 | f70377a104da | 759af6829ad8 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | tied | better | graph_prior_only | 40 | 6504f3e43a4e | 82ab34776a54 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | tied | better | graph_prior_only | 40 | 5968dd8eb1f9 | 75e3eb447613 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | tied | better | graph_prior_only | 40 | da73ca701ab1 | c4581ac84f15 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | tied | better | graph_prior_only | 40 | 11312062617b | fed610cbaebd |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | tied | better | graph_prior_only | 40 | c44afce0314d | 86973ca65730 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | tied | better | graph_prior_only | 40 | c4434c916fbd | 5dabd98ded76 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | tied | better | graph_prior_only | 40 | be09c6e3b6c3 | 3fa5f7406a16 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | tied | better | graph_prior_only | 40 | d62b786e45e6 | ee9bedbb41bf |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | tied | better | graph_prior_only | 40 | 849725ecd06c | 4ac5ff462119 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | tied | better | graph_prior_only | 40 | b6dddb3902f2 | 50991acf292d |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | tied | better | graph_prior_only | 40 | 07c6f980eb26 | 47ce22f97cb1 |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | tied | better | graph_prior_only | 40 | ed4b37f5ff35 | 7f873433484b |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | tied | better | graph_prior_only | 40 | e9fd8fbb634a | 3010d091f148 |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | tied | better | graph_prior_only | 40 | 9ea7ad377089 | fb4bba2685c2 |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | tied | better | graph_prior_only | 40 | 113bfb6a0733 | 5dc6f468722e |

## Deterministic Outputs
| role | path | contract | digest |
| --- | --- | --- | --- |
| readme | README.md | none | sha256-c5bffb2581f18b1ca62894f5ffac3b31c6a2562fee66e0035bf5d9c879a2d28b |
| index | index.json | recorded_session_replay_proof_lane_index.v1 | sha256-8a7ecd62b96747283b5db0d201e9916f28d5990699654c19d494160883006463 |
| summary-tables | summary-tables.json | recorded_session_replay_proof_lane_summary_tables.v1 | sha256-68b56591affe71d55838977cbac5bca79e60ccd7751b8e251288d5a7cd29bafe |
| pairwise-deltas | pairwise-deltas.json | recorded_session_replay_proof_lane_pairwise_deltas.v1 | sha256-293232b69cb1af292d64eed59fc04dc63c96115424958f769d8d40895c7cb73d |
| win-rate-matrix | win-rate-matrix.json | recorded_session_replay_proof_lane_win_rate_matrix.v1 | sha256-ad5e140eb1f28243c61ef9ad7a43c0687e18d6753cff76758a74e098c5fdf0cb |
| worked-traces | worked-traces.md | none | sha256-e77f40d986c22dc92cb8fe4e7398882d821e42bda1c0d6d64e234a2ba1e51d95 |
| generation-report | generation-report.json | recorded_session_replay_proof_lane_generation_report.v1 | sha256-9fb402dd5b6e3e455495530edd27fd0b55ebea2015c18e682d5c6a1f233a780e |
