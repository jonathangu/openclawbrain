# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31f15910a7f37f6942dfc1fa59eebabc12b733c2fdbc101bb92672de7f721f0c`
- fixture hash: `sha256-d3d3b7c9daea7f5dceb8bcbc7d0b182082662e4eea5368602c8cfc65a5234e7a`
- score hash: `sha256-5dd6f316c5b81798814268a6c2a2ad750b0181abe71785e338072de65485558b`
- bundle hash: `sha256-ccd98e20d08603d60ed0a1a9a759f77e5036444601fa85482ec9c8b689de0306`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eee7751ada66c814393120538ba88242a0ad04eb627a4b24f36524aa1be2a704 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8c16dd889561ef63297ac0fe08b9d87d52b2b234b73910970028eb955f3ef056 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6d2725ee5ebdb9d6aa89fa2892e3349f12ee60e3d8a513c84aa943f4a26e89e5 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-069ca2488098001d3acc92b90e8acb9649062b33f6006d552567cce72b6f7bd3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-17c56b23 | sha256-3c9214439b75a1562d64095149f2ba762c57422a6e57583e0d8dbf27bb50e3ca |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-17c56b23 | sha256-5c514864d010237b2ce544698ef49d296ff9ea610401f3de83dd70ed9331c089 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-30acd13c | sha256-9fd0ab28c9d768dded79ada39c621e60e385afad96764cf3e775f45120cbda0f |
