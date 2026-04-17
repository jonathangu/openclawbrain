# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b202c1c438845d3c1c73ddb7c1ff7926a10fda7c3a64127ae541d469c9475d5`
- fixture hash: `sha256-b48968b0fefff768efffea4ced309b4343ca39a6dbbeda150f150e0d012ef675`
- score hash: `sha256-76b91c2c0fcfe90b62c98e5e3bd94e12f03234345de7c3ae6d36f4069fa497b6`
- bundle hash: `sha256-14c2596ae990b1a46e909119129639b8d090456075ed412264d3de35f9f37ebc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-07841a59820286934b7db3a291f9a2a056f9291d9bd4bd106e744c3a6ac3c6f8 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fb2860c32aa6fb0dd978533ed415ea215a4abcb5873fbb1efa4cce7691e8fc86 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b07737cc5ba3a149dd99f9bfab9fd8e48eb31dbca3a2224a4f5bb8b873685fa |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5e288e9e8077627cb3569df38a5e527a928cffb35a82b8d201f854900ac2eb2f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ab6a3310 | sha256-6f1ad7e842c4110a89fcd93d131d004da708397cf0def3d3c6ad258f6a5dd973 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ab6a3310 | sha256-d8112ad6061308b96806c29e713603f09cae489ffac2a7e085ede0fa0f2b0eb3 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-aa9a8f03 | sha256-0cee06184a7d12b0b45bf508be75c54c7d64c1359c4c496bab227c1d53033477 |
