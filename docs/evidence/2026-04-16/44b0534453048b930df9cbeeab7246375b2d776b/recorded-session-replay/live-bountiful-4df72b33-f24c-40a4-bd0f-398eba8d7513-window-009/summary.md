# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-909da7829d18c7b4060630b0679e4b4f9d3623ad878ab660055447ba4071489f`
- fixture hash: `sha256-d6f02df1f7fd44472c7a5dc57cc2a6eccb52d15bdd1b99973bade05111901191`
- score hash: `sha256-bf511b12968fe8f9829a1040f83113db9d953ab07fcd741d0970d51311d8fc95`
- bundle hash: `sha256-ffecddb5ee66db562acc92f9bc15142e9698705d53304e24b2e400de5ce5030a`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-67d503fa6cc9652ac016339af3a8c1038900f3a66d390047a5f53eeceb6dc18e |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-55703e78632f00f4dbe4df7cd8349997bba784ed2dd93bcb4bfbf8ba896f95be |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-648658a61e236da1f73dfe82a0f5ad6a3ad9485fda37632fae3ba5c56762e965 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-0419455f152663de7c27215b773403d57e3c2e52bb15f4aed4be1ce749cc6808 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e943ca42 | sha256-a41b11b1d24ee39f70b6d195dea9f6fd2bbd70c085ebb8b63323d742a6260b79 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e943ca42 | sha256-607dc56650dfb2907e978fcf891998f7c5ed37d35a9104aee846288a57d9f7a8 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-82387257 | sha256-50e8494626e7e14257aae7bae55d3dc94db61c3ef688da7f9c345dbf0e30fd53 |
