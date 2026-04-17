# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3f29e706c8408e5460da5ae181547f400604bd45efe4b812bde36a617f82f5`
- fixture hash: `sha256-3bf32dcbf845b428f375103144110fdafde5982202bc1871fff67136d9720e81`
- score hash: `sha256-0102ec7c3b25abe423d5607a6ce7810406b3cb7c98000202894b13bd5b79dde4`
- bundle hash: `sha256-a86151e612139de77f13b9d187f7fd3ef913d210ce00ca5ec5ebfae09588af50`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-865ec1deeaa418471d6bb216a38e6bca377292e05c38cf14fb63e270894197b5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-32c799439ecfac5da4d546a7e0d786a09840abd235c5b05b792e4ac3d20625e4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-859601fe6c35162e5c7fad644354c254a846ecbcab6fd6b476f3ae254c7a8683 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5b7f74dc2840c08708844d80e41ffa91060e4e19603be4b79cd7b5493b3c0ea8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6db44d25 | sha256-7b0b6be7d53669d4b3a31705de0248711d7662eb94d875de7d940e0fdedfec51 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6db44d25 | sha256-db9aaffaf94622cb6339c2c2e6e277e7ffb8816fde46191c58b484f61e910b2d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-13bba07c | sha256-7c85d951ccdff9aa77e7bd152223501914aa7114a31cac9c259d3828357ce774 |
