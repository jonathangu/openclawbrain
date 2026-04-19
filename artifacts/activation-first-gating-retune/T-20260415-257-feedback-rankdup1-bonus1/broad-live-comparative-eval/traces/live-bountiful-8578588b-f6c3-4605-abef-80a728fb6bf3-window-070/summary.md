# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d996931dbdf0fc6eb1a041ef94d9b0afac8583cd9d36374bfb1b580b3ae115d4`
- fixture hash: `sha256-88843dbca5a068f5c1ecc181f00d1fb7032df4d94e84a695ddbe0eb2f4ef844a`
- score hash: `sha256-66836cfca98d18b822ed17b0efc3ddb5c41c232d7518f0f181b5700af73fe8b9`
- bundle hash: `sha256-8375c5217ac4b3cf5c4ffd316c80a85c05712c296d664420fc8c80c87fa9ca9c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b985231f563afd8790d84f55159160a675cf549c23dad2a3570253340699dd26 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-80f4158bbd19538f6561fffb2737ebe25dc7ed3d20d421ff58d194278f64b5bc |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-db61b0654a737653bfed285960b295cc06207752c5e87d79b8e924c5c8a9948e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-42d51791ad14802b390afc0f88026556726652518a30f9cf235d856886623442 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ccf30cfd | sha256-b72cb3d6e16d2085332582e617ece17d815e5da12b4036950e005a2897f251f1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ccf30cfd | sha256-086fe2270f77db404f7373c048b308092cc108fa305f7b261be06105a6f0e774 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ccf30cfd | sha256-b72cb3d6e16d2085332582e617ece17d815e5da12b4036950e005a2897f251f1 |
