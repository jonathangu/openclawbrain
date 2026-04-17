# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-56c5cb0bbf3fd4c3b31b5c0ab401ad3e4676c774ca7f6d545e285ace8c5c1fdb`
- fixture hash: `sha256-77236387d32f039002239433f6a8c01de43cc1e1b10880d323ebd379dc420a0f`
- score hash: `sha256-0458997027370fdbe83d85a2c72aff1ee810c31223ca10ac2c19100a1c0fb3c1`
- bundle hash: `sha256-3c7bf1a0abe5cbcbf5404856450cdc8e5fff20583fcf1066184cd9c92424d178`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4dec047b876b4ef1cbff2ba1d3926376bc0c710b4b08c16a2a7795d5ae337d56 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-23200362055abd52d4ca7c64fc18a7da29dbbc184e236761b350572f59046002 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec52be051fd8c35e6031652a2a71928c91d5d29570bf617df69ff7ed3b81ace9 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-82c2e9322337a51c2a8d78ead1237d1c86a2c371e9384338a573d04b00f691c1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-69155cfa | sha256-622d8c80e9dc089da4cf48cbb6f469ee57147fba01c0775f1238143fb2785fed |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-69155cfa | sha256-a60750e4dfdccc1ffa618e7f678fadb1cec90e1b0061b6b9a4c73f61c77685f9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-1d45e411 | sha256-ff7ca0ac718e6094fc535e94277b2b93a65d31bf9cb98944382c56fecacfca18 |
