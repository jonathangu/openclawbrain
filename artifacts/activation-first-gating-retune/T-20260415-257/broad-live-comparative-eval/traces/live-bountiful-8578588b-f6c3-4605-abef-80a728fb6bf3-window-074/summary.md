# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074`
- winner mode: `graph_prior_only`
- trace hash: `sha256-708994258585f9af49ad6c4184bbdccbe7e42b817caa649c9319156897755b1a`
- fixture hash: `sha256-3753407f1bfe4ff8110a80c454d28c5837a156f1ddd66c296964bb850a56a229`
- score hash: `sha256-9a38ee5ec1b1062c738bd418d44d9a495026342b0f8e3543305f4bdcb6bddc0a`
- bundle hash: `sha256-a2c45b56c2e37e6cc923e3132e5e235a173c27ea4e96e7dc1eaa56252e73d0a0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ee2a3369e811726bfb237ce35595dc08f1f6d73159670556f558d53a66965dc0 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-4ca6c67223e3c7052ded6b0788bf829491d16b084813733acca3581b0d5b0ed9 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-e5b05565aa57936433ac3c1e1fe32978b1438928bc278c77405a520daf88c8db |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-9604cbdea8deffafd692064079bda29d622dd3de62574b701d907cfe80376416 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-bcf6c26a | sha256-b05b2e06366b83979866f7946c0a6361ddd0ddbd847ef6df963d3b41651683c4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-bcf6c26a | sha256-b6bb83b6e1e8437a5d32ad66a56d3b2650a2d74d8b9b9ddae29309dbe76adc07 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-bcf6c26a | sha256-b05b2e06366b83979866f7946c0a6361ddd0ddbd847ef6df963d3b41651683c4 |
