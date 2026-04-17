# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-609a8ef08a0005ac8c6d28613bff0743081fda2af2229951c1bce5c2a71dd05c`
- fixture hash: `sha256-6a13457eafa6a8dea8911b77d2fb44eb3c714588ecda4ba2d46120f25504eae3`
- score hash: `sha256-8007391e80d88c08bb82226b6407753a60f26c7b5a2d4bff6c68aaed568a1fb2`
- bundle hash: `sha256-c17efabdbf0e25820a8f849496391fac36a116fc64a2eca49bdbbc3515760777`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-529cbc550c979df64d974591190e0c1d456cbd8f7265be9e27a0fc5cbc417683 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-bba831394b4af6a21930921cd5c0192c7ac9de683fd46813add842295e24fc48 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-d596cc5d22ff558cb9464a0d9f4dbc1b6e5f176c580a070060cccef9fcfeb635 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-133e50261c585eb187cac08d92c044408933b6f82ef34b8346b15ada31e2636e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-8f912a5d | sha256-32a1bf272f08542784418d7af48e6b1da95e65bd5eb58b9b2925551bec076762 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-8f912a5d | sha256-4299e39d7b6bc2e446b4c3d43110ec30b55036ed2e9bff11170c1d8f900a1f81 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e931bd7e | sha256-af4777a65d47a8d3922a3cadb0006f82c031332d01fed5c786f2fb9a3d076da0 |
