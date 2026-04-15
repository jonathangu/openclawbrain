# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1a576ab7fc82836d62896c5506ba892a7997f6c29eafb6387885075368088d2b`
- fixture hash: `sha256-e830bab1e1b5c601ab706b387c4f671be86f28c4ff56747b0f78265a86556170`
- score hash: `sha256-3b6d49ac344878af8bb32d7746d2f81a25af77a9c97acecb630cb68200602fb2`
- bundle hash: `sha256-66b859e3bd311893c55c0663feff2de81b3173c6ac51e890db7b6a23d103b17c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-309a76d5f65b7ffefd710af5c6f62a81606516631b55e10f450624750cad9788 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a994c5c9928eebc5c779bdf5c22cf8e6e985748c124257bbad48bdfd327ce2fe |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9ed3b93e98f84d59fc9ff79ed8b386d83d8c9f617227a8bce3e6cce25b733b6a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-19df49830bfb07ec6ac565f3dceb4371a4638228d5cc78bc367bf9ac74ccbd5b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-64ff961e | sha256-b050e5576580e03ff6593caf9e68a638f2977a09717579fbe8856e348f8ea11e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-64ff961e | sha256-de7e9d05b4374ebb3655efa54a75b20fa8800c0914b95c215d553e84adc48aa5 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-64ff961e | sha256-b050e5576580e03ff6593caf9e68a638f2977a09717579fbe8856e348f8ea11e |
