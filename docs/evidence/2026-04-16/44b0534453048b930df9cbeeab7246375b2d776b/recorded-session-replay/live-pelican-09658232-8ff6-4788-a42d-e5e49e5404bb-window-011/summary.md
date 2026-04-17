# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b2430ed58ee0abca0aa0224af405db6344da7702ccc6e754dab5dc0867b7727d`
- fixture hash: `sha256-0827a1eef5713f16e574a6c5a2c4721f6c9b9ebfe2794b2f08af42e8c07ece50`
- score hash: `sha256-616d5d85e5b3c31c95befbcaac00906b83cb8ef51d5044cb8a8c6329bdd2c159`
- bundle hash: `sha256-a14b7e5403972422c4e5e17a164f4d4052ba190f6a27fea29362e18fb30e761a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5d95dae3d2cb2e3da5df09b63b5296f231dee9a351a91285d6a68ab316bef562 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f12ff5ab4b71554a58850c7fb42ecd8be3dda14dd3131b6f8be7e33a8b6c520f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7a07c19d5d1d87f92005631d38b136cae4000a3b2ac5fc1599b9f27c21eebf16 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8fed9d91466278305b53e9aa9edc52518cefe33e537ee2d4f2bd357924f4ffdb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e51e864d | sha256-0bd03546f051c159f16f2178d4f6d118fab7d0b44ce371df3d496fda4c892902 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e51e864d | sha256-665355a98ed4b6f3368cbbff5553fadec1dbb0313bb676687df593fc4a3482e1 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-af0ef950 | sha256-a6abf0a99845c11c1ef8d9729369eac64344827a6985058863c1274df67f7878 |
