# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-db3086ba9270f5f47434d4a3c708e73ff2624adb056b71992e75ebf839a91592`
- fixture hash: `sha256-2a8d321cab2bd435ac998d63d68b17b7fce95e9a0ea6d02ef75e09676d4240bd`
- score hash: `sha256-853ace4dcd280c6da4f0380db98c959497cf275810dae147ee11074010dd6646`
- bundle hash: `sha256-7f2c2424adae4f4a4be8782a63ae90cccb53b56cb7a51754a9b316cf643af63d`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-da6633515ce74e28e9f8bbc2cc587b6b0548deffeab6470c77c67fc675828106 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-646ff47b0c7ae12ab9f0c426ee4a514d18ba58eeb71529b7e6ed28da02b6f2de |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f4226fcbdf09885b4977efbc37e74b04ec449ad98e2c99abf47e2eb8695b8cde |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-95ba0b602a04b7ca9afb450874811eb8fc1d59e2181525e46b86388775744c98 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-65f45dd3 | sha256-ae75852a707b337b3040fcf07483cf188bdf74710b730b8a26f1817a8bc96ce5 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-65f45dd3 | sha256-21810005ac0e5a3cc9320f94324d135455afc800d29f812b20abc1d9921017dc |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-83184970 | sha256-d0304d00fdde8814772f91a03a9eca4faee6e0598bb9fedae3050dddb5aee4c3 |
