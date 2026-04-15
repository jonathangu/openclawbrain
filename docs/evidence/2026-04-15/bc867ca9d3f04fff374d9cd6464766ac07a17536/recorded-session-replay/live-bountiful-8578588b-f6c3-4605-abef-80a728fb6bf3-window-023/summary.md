# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ca0eaf6119fd5eca026a2af3b944ea416d6b1efe4eb9fefea42b7c6ff57e6bc9`
- fixture hash: `sha256-3b94f7345ae9ef307ded464a5d75cd2634838ce66abc47afb09a10bd7f7fb2ad`
- score hash: `sha256-f7a3e1da4119e0b3d409e977a48a45bebf6a05944ab0d479e539e7e16820e874`
- bundle hash: `sha256-aa3232c18ebb5529c595256d9f537c05f97ab14a261f6204ede2f3bdd486bd41`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9867396f82c81199e6f66038091f8c07ee4ac9568ec27c50c752857517ba3f4e |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-81b8b6369e596da6d8778d7d07e0c4545a6a6623333801000c789055f862be6c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-70fb65218f41c204e89623e8580cc571e6a9ca26fa91b8770739e549f2ab418f |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-36845120323bcea2919ecfa94f600803f88f970335770b33e4b3318bbb1fdded |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-19551b4a | sha256-990d39ecb39c85b3e7612e7c056c9f2aebe49b2631573243bc8cdb66969255e2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-19551b4a | sha256-5fd4cb2c502ccf91c8bfc62d8e1a66fb536233dc7a16e96c322838d8d9c350b3 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-19551b4a | sha256-990d39ecb39c85b3e7612e7c056c9f2aebe49b2631573243bc8cdb66969255e2 |
