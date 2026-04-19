# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d8a6a685e755dad1fa0654d4b6b59fae51d32befb5054b1345ad7441ba8df43e`
- fixture hash: `sha256-6c5d4c7687666f33983698e226b698dfe912054eada98b448401e6f4fac93956`
- score hash: `sha256-f063ba89e56f88c24e839a0fd854af8209f670f0c0193f87dbe9738a008ef937`
- bundle hash: `sha256-1b74bdaccd007b211c9b8500cc84de41f699a2ef3b6a7514733f11874eeccc2d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ffc68f5afc82e7216af198fb6c91f17aa507e3e600979abda3e7dedfc7ea0fe9 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-6ca376fbdddd6aef46b9f68bd81d2f599e7bf7ce4167adcc8f54e20b3247cc6c |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-be103dbb2e270d462f08b976b227058da814b78840553df0b8da9773ed5dcc39 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-007649168dd5420649fb6e9d21b663fbf1742dbbbaad71e1015cba41c1024755 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-bc0ea1ee | sha256-613fb82b4b8db23999e518ce5d0d825b90275f2a6bc5a81cd7be136d745211e9 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-bc0ea1ee | sha256-08a841f9d8076cc768ec33bdd0fc4b08b71b1f87e852218e56c5741ef8b46235 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-bc0ea1ee | sha256-613fb82b4b8db23999e518ce5d0d825b90275f2a6bc5a81cd7be136d745211e9 |
