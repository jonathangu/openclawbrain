# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-874a83098560adaa94c38c7c63cbf4c86efe4c86090d606bbfa34849e336a8c9`
- fixture hash: `sha256-b06776d862580d01d558132918aaffc22b9130c1387f99ca2438e1c6cbf7e22c`
- score hash: `sha256-9af4e522e1582416113d4da439bc659b0fdfec08a2b19ffb1a2d131d3f4af409`
- bundle hash: `sha256-6b6b894bf1c6a6cdf84d5a5c3ce18439110466258c5edcc95121ca7f8c5f23c6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-74c569833e5bf27fad2f2f842fa8eaa7d60bb320f690bc493bbf6c394f309f6d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2e31558c5b6dddd83112e7a35bd2f1c17e923e1d69dc03e95bb478603a38a554 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-de1f0cd4eaaafe3f2d40dd75ea39e48ac56204f9131b2064f1fc44888426385c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-260ef643e49260624a11184e968dcd137b1d95b501fbee6d6da041fcd7b2f2b3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c8520750 | sha256-c21dc809fa42d7382e54f769e518b2c4fc74bb4bcb3257086407696039817587 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c8520750 | sha256-5beaddde53091f85b72f3979b5f7531d6e584c7e587485f95871517654164e8c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c8520750 | sha256-c21dc809fa42d7382e54f769e518b2c4fc74bb4bcb3257086407696039817587 |
