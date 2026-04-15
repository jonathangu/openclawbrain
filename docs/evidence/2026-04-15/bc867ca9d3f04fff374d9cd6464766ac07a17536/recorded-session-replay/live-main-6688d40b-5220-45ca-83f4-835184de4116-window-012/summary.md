# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3f29e706c8408e5460da5ae181547f400604bd45efe4b812bde36a617f82f5`
- fixture hash: `sha256-3bf32dcbf845b428f375103144110fdafde5982202bc1871fff67136d9720e81`
- score hash: `sha256-bbe70bdd54d4c64ee3bf0424409dabf5eeb68ca543d9f56ae6727ce3cf3e6a40`
- bundle hash: `sha256-bb2575e1152fbd823bfcdcb9e600d827c308e67badb1950ad80b6f44b333a438`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-865ec1deeaa418471d6bb216a38e6bca377292e05c38cf14fb63e270894197b5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c400c86e04fdb8a38496b5dcc9eb03c1e7fd26abc88d50af5ec7e295ba4c7030 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-66d017428eec6a0f680661fa3549c729b9ea6796bb9e4be67117190c4059c60f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d14243d00c756a93ec4f94bbb7f6fdd0b20999cd368ebca251130a61b531b3e0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-514e76a7 | sha256-88ddec36109ae8690cd32b89056e94df3cbdac8a76d06847928f35702e1fa278 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-514e76a7 | sha256-ed20404b6381d96ac69ea8902681fc88b4a66c0344bb2e624d191af6e8ff4453 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-514e76a7 | sha256-88ddec36109ae8690cd32b89056e94df3cbdac8a76d06847928f35702e1fa278 |
