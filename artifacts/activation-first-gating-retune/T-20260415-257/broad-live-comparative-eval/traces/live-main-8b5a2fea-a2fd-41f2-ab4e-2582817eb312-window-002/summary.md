# Recorded Session Replay Proof Bundle

- trace id: `live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002`
- winner mode: `vector_only`
- trace hash: `sha256-e0e56ffd1c26d20085e7a9eb3248f58dfab8c43d92d6bc35e804da203ef4f7d9`
- fixture hash: `sha256-e4b8d39277cb985d3e9ee559f9e373775182720bfc10b6d9350141f9c5016460`
- score hash: `sha256-61eec6489a6a6726c55a1c01595cdd23cf413d8884dfa07769d202c40138d274`
- bundle hash: `sha256-09d597cdd081ec503550095c728d57416da3a9c95b600f24a2677f0afed061fb`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 80 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0bdf6c0bfdc77dfb35df2ddd80b080b8e6bbd2f8f1020fedbea4770e769e1c72 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-35cac709efe0d3e6a9cf9b67df707828778147dfb70a772541b3a42a53272657 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fb061b4f2dd9092402b84c1bd8c754a311244c8c1432639712b4cce69e1f7361 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-fc99ebff7aaae47cb18181a4b48a3aa999e163ef6abe3438ba64a38ce40a1986 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-fe7c8f72 | sha256-0f7ed7d849c7881b60a0521e64a05c53354f27657cde7ee9d6c64aa38811ed31 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fe7c8f72 | sha256-cceb7b3f264790d6eb830e9488ac1eb92a83e495a73057e5e7b87eda6bde3514 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-64a7562b | sha256-736943aa28e822a5bcaa311c2af543a32f0562d1f994934c642e07e317f5ffe1 |
