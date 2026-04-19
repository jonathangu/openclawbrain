# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f579c26265d9087760a95275a1ed5d3c29a7fa2a5745f0cd5985ac21a42da923`
- fixture hash: `sha256-344c0f8fa42bcaf494090e8fb4c4629c475783bea29bc527602ae9b6d23e9791`
- score hash: `sha256-429f4f8d1ab175ea22a1275d076de4c61a8711f810b1ceab86ce0c64fee429cd`
- bundle hash: `sha256-ebf02dc01e2b43a511dfa6959f246da7673988bf52e2bb6af5a13e3652bceb63`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c8dbb3069efa791ffae92b155399c68bb15a7550a719f9b3772c99bdfa5fdc |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-990538d03ae74fbe025ff2dcff48e3fa3f275754042821c42c52f5c1ac0fc512 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-43f755843df52e32153ab4752ceea62c4af9cdc8e231b3cc9b1b68798446483d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-70af010efcaa70fb3e1e87fe9e2bd660a2bb5e8437cb54377c100dd42e8f413d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-fd5e42b8 | sha256-a4aa744f9ceb533b60e538413c79c78776c258bc8b4b93de75a41825af05b7e6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-fd5e42b8 | sha256-102798e11e1d044e16a794ab2f444d4ffb48bdeac5189ef2b1080fcabd52862d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-fd5e42b8 | sha256-a4aa744f9ceb533b60e538413c79c78776c258bc8b4b93de75a41825af05b7e6 |
