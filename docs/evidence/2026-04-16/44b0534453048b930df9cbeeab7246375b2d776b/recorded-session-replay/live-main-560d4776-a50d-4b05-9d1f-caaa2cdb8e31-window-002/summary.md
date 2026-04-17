# Recorded Session Replay Proof Bundle

- trace id: `live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-035806e6f9bcb3753f58456b70d56dc8a01f4abf60114aeaf359384806f6c24b`
- fixture hash: `sha256-74bbbcc2ba3e23b87dadc56cde438b46daa30c3743245ccd0b40d24de1249370`
- score hash: `sha256-8adced6f7b2dbdf1d1201965ecc3d841f815d9b2a60b89790ec2559ad5ba16fb`
- bundle hash: `sha256-951939f7d888061c2682626b447cc0ec8d56b0c9ec21a4cf5850661dd11e31d5`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54ce8361598e1b4080ba115badc91e906441ece2076bc91bc1f9f28df2706034 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-56f0ffa79020620ae4255317e9394a770e1f4c0c2fc67a4a636858dbeb582f5c |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-2240cfca09cb4ec8b3cff2342377902afe9df9bee5d51e10bd1347af5c93bed7 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-266f2b03fb98f9c0d17ae74d90fa12ed2691c416a8108ac065b302531f019548 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-de9fc00b | sha256-1b2849432dffb9bff1355306ad63823eb1109f0746dbec5e13be97aca515db0d |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-de9fc00b | sha256-6c002264badeff9ba99f8f66165e4e92f15415561f1255ca4e53aa0d94f23cd9 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-8c60992a | sha256-dbf9f7a36a5024d2f9b0ecf6127b9104a2350f59d3394d447b582b120ad6c528 |
