# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ca0eaf6119fd5eca026a2af3b944ea416d6b1efe4eb9fefea42b7c6ff57e6bc9`
- fixture hash: `sha256-3b94f7345ae9ef307ded464a5d75cd2634838ce66abc47afb09a10bd7f7fb2ad`
- score hash: `sha256-af07c2ecc7860f97ede78f5f3e989b44145903e46187dd4462b59408209d3188`
- bundle hash: `sha256-393887aee2d6d130ab15f74748134afb74e4f0df1087db671c36ee7c48c7c9ae`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9867396f82c81199e6f66038091f8c07ee4ac9568ec27c50c752857517ba3f4e |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-81b8b6369e596da6d8778d7d07e0c4545a6a6623333801000c789055f862be6c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-70fb65218f41c204e89623e8580cc571e6a9ca26fa91b8770739e549f2ab418f |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-ed4bbc1ee9bc636f5176ae513661d9a39982830a1491e23e99e232de9be441b3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-19551b4a | sha256-990d39ecb39c85b3e7612e7c056c9f2aebe49b2631573243bc8cdb66969255e2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-19551b4a | sha256-5fd4cb2c502ccf91c8bfc62d8e1a66fb536233dc7a16e96c322838d8d9c350b3 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-ab790447 | sha256-da08e3421a5b85ea570a51071c55b7c05ec30d6c140c5ae2c94052354ac99191 |
