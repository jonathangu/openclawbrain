# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a9f3db5c5e9f18aad5ca8aa8c8134dfac254479399202badc35306faa348393`
- fixture hash: `sha256-af03997f06ab50c99afcf76923b04c21e1338d145564c582674e59eb816853de`
- score hash: `sha256-74c8423c9ea824c293dc22990798085ecf437996214385474e0c4b15f7dec3b0`
- bundle hash: `sha256-6cd764fa864100a6d052effdc329e3ea331091178e4f34eb0ac02c5e22d6da41`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-048408a5bc1e1d56a6cc83e227b9a2958b83cb861b21925fd209ce4b8456f636 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-701851197222db065ea5de5faaf0f1e513878e80f2db87a2ddd592470470e67b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e9fd99d60e83b389f8813c8056ee28b22cd4232d88cd50e21137450029016cd2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-624ecd18cb792dfec41693d36c98e071aaeb4725c8ff38c487681815654a9ad0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e6bd646a | sha256-c8a15dfc51b9c34b8cf8282a1d05f452b86a3527827949050d6637eefd54db02 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e6bd646a | sha256-a9c8425d8caf627971890221a1b6a379a9155aeafe721af4f51c80d056c3eda5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e6bd646a | sha256-c8a15dfc51b9c34b8cf8282a1d05f452b86a3527827949050d6637eefd54db02 |
