# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-1df02420998a45ff18b6fa7592e1d6cd553e69e00670f629819f48f156232f3b`
- fixture hash: `sha256-d26e41f8ffbf777f72220318ad80ec7f532c81cc4e8c86beb0f89befd769d272`
- score hash: `sha256-a5c102cb45541b25eaa1031ed15842d1229ed8fb5484ed56c77a5d51c5ec0468`
- bundle hash: `sha256-04c0312980aba34b93685cc233e85daae069b59dac30c4234f959784a5af41c6`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 7/12
- phrase hit rate: 0.583333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-caffae64068969fa7e1d950417642498125ccd7a52b99fe5538a0a0e555ac8a8 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-342ba85c5311a47d1892e4a823de46a4b8dcdeb9ee298ca676be05c4132f37bf |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-1ceb4eed1f20b9bdf62734e86d9bf91775d71eeddea33e9a7e7f0673ceb1ad36 |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-356bf1445d9ac1894991f0c0d6746d0ad86c919781ca0658eabbbe0f2d387702 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-29ae38a4 | sha256-4f7535ce6116ace6b204e44285df3439234eb752bbd5627f697a49555ec83b4b |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-29ae38a4 | sha256-0124e6b9f6b78de532588325242dc61839c8526a3e2ea57d84aa65b48e46b051 |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-696954af | sha256-9b9853c699f97b71a0795df585e516f1d637885d02f19cfc0276810d57e7e2a1 |
