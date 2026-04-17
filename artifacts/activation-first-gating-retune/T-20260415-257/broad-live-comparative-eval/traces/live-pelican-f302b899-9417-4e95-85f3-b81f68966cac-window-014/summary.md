# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a76e30e78a3c6628f4a84b691210152bcddf9b8fa0661b16388a6ca59daa23ac`
- fixture hash: `sha256-2f161d785d6fb80ca3ab0af035b3aa3abbed725f829a4bae1a60b67e83a88b19`
- score hash: `sha256-3d215c5db0076ef0a0191890fb8c3421c18634b1dcc685780060a88a27bcb7b1`
- bundle hash: `sha256-c78e5fb2c1436d0d156ae3bd228e59d0a8cbd16863226ca963e89e5bbff9fef2`

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
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37705891572c574b5f2ed2ea56d6ec8c0372961de1b290750eeabdf9bb9948c3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-be6a1ba1ca54fe5002ea996833a82978743ab51821cb8b78d0bff5f79adb35a8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bad4f4f76e630a1e454066b0ceca8b46af674f86bde1ad62a123a2397ac28d29 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c7032c3b8dbc40f4499b9b9294b71d941241bff28e876ddb8d11b427aeb4adad |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ee9e5661 | sha256-2caaaa34e5812c17fcb9d5ad5ed6d7eae2a46bb51fac3e62978e87fc54b41f55 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ee9e5661 | sha256-92cc3344cdc6787893fe8f0a45a0f6933eaf23789206372d78a08b2cff691a61 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3854cf70 | sha256-0140e72e6a889ed91c06561fb4bdc569160cfaff291c6058ca59bf717de28dce |
