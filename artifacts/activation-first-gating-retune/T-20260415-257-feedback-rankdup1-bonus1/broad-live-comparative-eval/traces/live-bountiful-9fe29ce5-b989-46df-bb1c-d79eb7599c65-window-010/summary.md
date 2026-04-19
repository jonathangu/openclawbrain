# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-47e2bbb0e3231912cba4ca695ee40afa634c254d39846db31c816e34ed5fce09`
- fixture hash: `sha256-3d564635ad5110cdb42d334240ce415d162e817662b201fa3f2e0f2bf21b9556`
- score hash: `sha256-1e25862836b2f451f8a5aaec5f1eaf6715ea360d04f65c1cce2fe6c8c2b692d9`
- bundle hash: `sha256-a93eac727959e3deae6c13f9c3b64d739d0c75598ae9bc13d28528de3de4787a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-caaab5b4f02580bfcff0412720641649b1a6015360a0de6e547e4cce40c83035 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-89c1d440625dcef61a81146c4dd4b3573b885a9edb56ba2c4a3fc7c4c9543eb5 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-947a89027463e3da00980d543c1f78ed6d4045782010c0a1f011e5dcf312fcbe |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-67bbd27589cd8025565e70a5beae600415ce9c0f6b41cb20bab352deadf63362 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-17212974 | sha256-b094640f2cf69c4ca8a13dbc12059a97699cd843b72559f538863939d5b8be41 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-17212974 | sha256-a962840a319a72d6b7b2e6a3cea02ddf4f1a6ccfc9b98896a870b565cd3717c7 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-17212974 | sha256-4b35f3a5a7ea57e9f910e3bd0f637b07236d53abe7d7a9472556d343f69972b5 |
