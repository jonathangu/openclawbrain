# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-970ba48dfa6c96d0a4965b4677af4fd629fef3cbc40e01188dbcdc91cce4557b`
- fixture hash: `sha256-be39fb4084ab4014f594ecf827b8324c7590b1b3c6ba2cabd9bff2dbd9a1798b`
- score hash: `sha256-23807bbb8762657c8ec5b3640031144d47b9c4f333bef103a71fe7b8e41d16a7`
- bundle hash: `sha256-652480aef008c76751b5549d94bad716acc56b728852a61316eaa3b561a22b63`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a8c8fb966bff98fd7248d900de12653a4c0149cb3145489937f87d5ed585d1fc |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7e720fb4871da559a9e18fde135176eb059b4f6ab397a9f676257d5581add23d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ddb00cc19ca84432283b180100d430ffc72830b13586aab3f91f93881b3c5128 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-ee727ad623e0317649769ad38cda1385bc455bb3db588b3c67054e68c7b8eb3a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-764d510f | sha256-b2e03973cef62661d1e15fbd042f682ebd0c03e4e5ac5f168810323f1f37c218 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-764d510f | sha256-8e9c3685995c541cd0872ddd4e83dcb393e29ad76c3d7222387103b9c12f759b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-764d510f | sha256-b2e03973cef62661d1e15fbd042f682ebd0c03e4e5ac5f168810323f1f37c218 |
