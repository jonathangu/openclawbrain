# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5b1ea0e5b592ac9f44b7a4352ee387bb879a764114dc9ca28f8777b2e759540d`
- fixture hash: `sha256-64d81bc3848c97d19d6184af82f48df39e39a81124f70e1ee97b5963809c5506`
- score hash: `sha256-400b0c7238359d5473ed98a63a8d35fa15ac1cf779e627efcc38726844d866bc`
- bundle hash: `sha256-c763c7fc67014a02007045d48fc35132c940e646ed5dda330b18c82b1c9e22cc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-71a0f39a2a308837d02e7c312f0b041017358409f7a91268e26a3ecc203deac0 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-d45d411f36ebe58ff34c84fc175d59c667fd8c9803a840a996d6c4c881774468 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-57951900ef1e5a45129d0640ea47c2b0eaecfffb5fade8d7cfa82663954624c8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3c2c91f7865f6d550ea991480403eac8ef588431296597c051f349168f963978 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-efd8fca8 | sha256-8ece610fa4e929c5c6f033ffbf87681ef0df3f101e882bb9722e3222d92e882b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-efd8fca8 | sha256-b94c666f0b7c22f0dc81dd45bd9ccf435950785535455468e2bf2ee882019ae3 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-efd8fca8 | sha256-8ece610fa4e929c5c6f033ffbf87681ef0df3f101e882bb9722e3222d92e882b |
