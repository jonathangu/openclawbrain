# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7c9bbe1bc32703bdc0ba57cd7c2e5ba0147d232db874dca18f8d1c93a644936d`
- fixture hash: `sha256-1632c273e7fcb25c5de9fdb5adf5c07fcc4c43677737f0e63cd97217f3d6d9e5`
- score hash: `sha256-d533aa8273969dca401f9c2f5a60312f13e86610a185d9a402e7ff5107ce838b`
- bundle hash: `sha256-f78442b6cea3dc670b31e1bd586f5ab2c923afe4b092004171d600b5e738f732`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-80e9af933d3a18c2836442131236b812d1fdf8db3bb96c2fc77c951fce5a2ed4 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2163aaeaadd09f1325905013960449c7b5f746b5d5a288947fd19c23579366b8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b357255728716eed982002f73f8b22cbb58a5ae27148ac12de6e37d197d74b52 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-f393604e79d0245378d93f042c647d4c37e5040f6f35eca17ebe36694df48daf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7356ceab | sha256-9902127e93116d4046268e1747cd1081aee1feac8f74b23335ae8c3ae0e4e039 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7356ceab | sha256-557d3d1b88ac13807027e957443ec1db18e404bbbbd35aa650d00244d12acf69 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-caf91e7c | sha256-deae112385cf034d23ff8929baf28a002755b8d058ff8fec902b025ca6d7d3ab |
