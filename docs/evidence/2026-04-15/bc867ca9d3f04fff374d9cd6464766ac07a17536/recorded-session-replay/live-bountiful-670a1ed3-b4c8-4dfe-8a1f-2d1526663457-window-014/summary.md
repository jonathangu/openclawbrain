# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-609a8ef08a0005ac8c6d28613bff0743081fda2af2229951c1bce5c2a71dd05c`
- fixture hash: `sha256-6a13457eafa6a8dea8911b77d2fb44eb3c714588ecda4ba2d46120f25504eae3`
- score hash: `sha256-de783da0542155dc3ae1fe2341a2bb1b186868621e388026aaf8688841dd8f6b`
- bundle hash: `sha256-c684145354e8f2def420b70f03c1a8c66511def95cf494be3f60fcf625d5f7ed`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-529cbc550c979df64d974591190e0c1d456cbd8f7265be9e27a0fc5cbc417683 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1d797d0bb5fa6c6d1ac6cc25ebe5a66a6595b5db9cb0151983967d78dfdb09d |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-73e1d1b91570d54b9498ebc6bd10e63e9f361b7d96220d9916b93e3679d58ca7 |
| learned_route | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 2 | sha256-4c873409ff317da42d5fb90197416dce1eb731226223da3c5a3606180241d9a9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-15a3e15e | sha256-0bf464c282f6810a92a6ce6da6dd32247749b7ddce474bf7c1fc39b77acd9e5e |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-15a3e15e | sha256-c2eb9bc5c82a3afe117c704bbec36a3a22df29852f6373f2e4f34469d7f26e83 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | no | pack-15a3e15e | sha256-0bf464c282f6810a92a6ce6da6dd32247749b7ddce474bf7c1fc39b77acd9e5e |
