# Recorded Session Replay Proof Bundle

- trace id: `live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1238bb817085e52d5386a747baa6ea8bf61e3a37516af898c3b116b0246d9843`
- fixture hash: `sha256-edd92cf0e628f6e0582722d507204fe8af0abb5e8a70f6ed2001e47aa93a6a45`
- score hash: `sha256-329b0f1de1e24aa4c91ad41ea6fd2b147fe175136e38da0d61a67938ce318957`
- bundle hash: `sha256-d034ba279bb9e4cff69223e973c48de5391954304a32886b3051ba4e0a045fbf`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4854c1e8bca255629a7b6a7159c3b8aa6429b304343bbf531069be2796243202 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c1e2688456dcc1b3a3f51f7ff58deac342e80a27e801b7278020a56f94cb2230 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8f6cfbf8a0a1cc80c28eacf177ec7a45b8bf013cb3beed55338538cf8331ae00 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-23e93de4 | sha256-8404d1b64e05000e56063f14a28e1c52d2bf47e198a157705abbea2a0653ae8d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-23e93de4 | sha256-90f7a101df3c68fddbe4bf43fb0d27edc905050cbc0a9e6449be27c0985eba24 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-43042f03 | sha256-721fa3c0dc10447ab76408ef8d58d0fdef3dc1c0c70270584c57f909e51f3508 |
