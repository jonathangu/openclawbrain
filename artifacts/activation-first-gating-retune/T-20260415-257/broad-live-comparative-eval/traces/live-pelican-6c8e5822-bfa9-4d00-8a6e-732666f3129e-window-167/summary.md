# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9808f8fc2b34b9bdd2037973a4af69235ae13bd43fa03c06a9cd2c930faaaa29`
- fixture hash: `sha256-c59bcee0d7e5004e8699b7491ca609cee1baf21baa2824b5dbd8c966b365083b`
- score hash: `sha256-cdd3fa2919b724e2a8ac14a8c12d17ba2275f46509b8a7326aad7fa3f073990f`
- bundle hash: `sha256-534b5edf2c0534d2810e9e9db6dc9779e29a05f04df89bc285379eb20b4ca411`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68b72eac2031bede9aa8770eaa1f000f5f4b3a15976311e630a954289393b0bd |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca8e6e1b738ff095122a05218de2d3d7371868754cf275c142d1c08d4fd9d118 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-bc63d0ef30a240fb5025909c410e676271b09bc67b40e571ad9e0754f73e8587 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-f31ef29c08e4b23eabc184959d238fa65d520c158ed3102c5c59612ec063b7ea |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-298463cc | sha256-4b56c4161d3b0760fb690914d60537e74d0d6d2e8b580b4ef29272894501abc9 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-298463cc | sha256-fe75114d298df60f82d7ad3d91720ef1b271868655292eae4d2a81f0557a5b40 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-32796a79 | sha256-d2412a6d99ff46fac3843d0f9dbb7a92854e8095743e0ece68646e855f8c5438 |
