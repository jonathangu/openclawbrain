# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004`
- winner mode: `learned_route`
- trace hash: `sha256-414029967a4dfaeacf3048f9cc246c927617fc5206e50ba6c1c2944d9dd8d93c`
- fixture hash: `sha256-2f96d4d80b85de0482bdf816d900c02ecf0137642687879ce902112bb8056ccc`
- score hash: `sha256-ab88f1a6f6fd75f8d4acd59864dfdd50e5ffa8090d02514c0b679e8bc24eb6ff`
- bundle hash: `sha256-b252e697b03e463dbec3147b9b46a6e88482b84648d502afd916c61105b6a763`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 60 |
| 2 | vector_only | 60 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d02c67e07266eced41424ec8d8650df73f7c0173cd9e14609381c09dbbd89d1f |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-40fedddd197981104f8cbe1286e8fa73ce736e57c950ad3e11f28b50e5d5e03a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb1254f059ca4ed63b01214f798dd47c60fb72de7e13daa40c8b153ba2868336 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-ec31e4d38689ee1e192f72346739689c055596f61a95ac6c303a9096658bc1b2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-4633edcd | sha256-823ebf41a2586a24eadcc9809baa66d5d09b3400aa710937f5591e92621ada88 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4633edcd | sha256-1f25e1ae66756bb8bf7e7c81fccae61191fe76b3e371cf8d3cd1772f96db153f |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-edd04ec0 | sha256-81b3eea074bff1e72ebbaca9d570be3160f0311163baec66e745a7bb71bded9e |
