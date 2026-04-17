# Recorded Session Replay Proof Bundle

- trace id: `live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1238bb817085e52d5386a747baa6ea8bf61e3a37516af898c3b116b0246d9843`
- fixture hash: `sha256-edd92cf0e628f6e0582722d507204fe8af0abb5e8a70f6ed2001e47aa93a6a45`
- score hash: `sha256-2a175c6cc54269d033efde551da5d0036d5030ad2c4741b3c633d524e8eccbbf`
- bundle hash: `sha256-91a4b3ef7b6df158ed7936f1883463c87f5ecbf8cafad1ae7f922c8f0bac7990`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d2f9bcbabb6e41c0be690a68df09ebb71d4f854521659c85e60ae6817b1b9042 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-db14294ddb3859019ebba6ea95476704434a066772ba27d844332d5eaadd52aa |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a2496443efa9e6f3c7d812bb92b06f86f75c1b0719cb241499df786ea27a91a0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-592f2f9f85177125acbc45c1da2fdcf7077e38859ff8de00fa5f1ab526352002 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-98b39183 | sha256-8c9dd40e8205b6ed77b000d44538987ab3ba88915f501bf8677190e27d373798 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-98b39183 | sha256-5fd3dc5ba6b1bce6c85919cb613b625a25927267b72b70d31ea6ecebe235017e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b7ce82a2 | sha256-35caab47c4c6d3dad3b27b8ae9400b7688dabe6e98218b6b01b07d5c30670473 |
