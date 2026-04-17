# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-512c430e649faf76044870db1348b61987384f6c4b42eb2624038c368ab6a4bd`
- fixture hash: `sha256-6f2c5641408f7a03798669e19a288492bcf8f6f0b8043e459e2c72b4bc2ef9f6`
- score hash: `sha256-e01712082c129f2ee72c21d6a7191cbd8c4a175e48af00578e1cb617eee12b61`
- bundle hash: `sha256-91092ea90dc1e670725c97727bf67540a29528a8b7b277847aaa01c2692685e8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fc8b4083586f10fbbdda0686c1eb4cc964fe1c89c35a3824fb52431cfb03e36 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-657063503163f55101a0cc52108a8f9aa7b2f440d918f9e0f94ecc86b19de2e1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6e192852f5afbc62f2d8cbf19941c29441fda7d7d47b687055b8a562e55de1c5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-10db9f745e7feed9a678355ed7db1bf3f8efd126ba0748954e4c76a2ff745bff |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3558cfcd | sha256-e134d11358a169f6b19a4123f349472ebd9414f9de2615c4653817bf3225b5b9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3558cfcd | sha256-da74eb04029c3380133e65286648f47209f9301678dc4bd7c472d4795d176bb7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d206cbe8 | sha256-8a73fcd408e6c0b0c8c8710547efa19d556c29b8ba5f2f4200ca7d9ec7f16819 |
