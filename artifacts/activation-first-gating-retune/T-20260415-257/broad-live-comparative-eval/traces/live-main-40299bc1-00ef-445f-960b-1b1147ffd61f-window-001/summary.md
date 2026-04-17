# Recorded Session Replay Proof Bundle

- trace id: `live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-906cc40ece3b0fb4a531e168af49c80ff11ea62f07a752db4c8924f98d189aca`
- fixture hash: `sha256-565a616fadde1db10f7ad35acdc4ddc02cf8260e0bdbb94b6efea52c6bd1c593`
- score hash: `sha256-db1459cae874190780b05b51493b67f1ed84037fa6e81f4725493a69a3ced2ea`
- bundle hash: `sha256-026704896defae3b47c05d29e7c91195b43e72f528f7d4be7af4138ed5a2ef3c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-faae0cd9c1a72b5cddb9e2597356cbd0162b3076b86d83b9efa7f531bd948257 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8d5dab5e98f84b691b7d3ed8a7b230850ef9b7429292929fd9c7ca7467e1e7a2 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a360440a7b7d7b555987c4b5484e7697eed42c765db1a009890c446cc372a58b |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-5a04221ab7673e2dc5c66677c8dfefa64a8a38517c384cd2ac7c652668330e77 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6a5fe3b9 | sha256-899f8ec7f86a2f45eb5e97d936e43431bbe1c4e581f3946fd5d8cf1a67a76ae4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6a5fe3b9 | sha256-32214dd7b42512c146a2740f0ebfe3116645a89d2c943165796b5dd7e16bd3b1 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-d2bcebcc | sha256-df3ca4becd6060bf9979268561734d9043a3259dbecb172fe683593ee28d7998 |
