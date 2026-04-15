# Recorded Session Replay Proof Bundle

- trace id: `trace-correction-deeper-proof-story`
- winner mode: `learned_route`
- trace hash: `sha256-34e695185a4af79eba4d4526f41b23f1e694980113c7bb70bca61b4149f2d707`
- fixture hash: `sha256-51944764426fd1dd7985c8105e13534bf617e449a4d2c80a2255c1e2a25cfd0b`
- score hash: `sha256-f53ed5a5c8ddc917c02949b7670ed6f9536f04525a72db8b54a2992971b6a2ce`
- bundle hash: `sha256-e7d86160a979331c62d729913749bacdf054436b9ae91df89d3b78af0e8425e6`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | graph_prior_only | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 8/16
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 3 | 1 | 0.5 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 2

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 2 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/4 | 0 | 0 | 3 | 2 | 0 | sha256-f751ee87184d52e0299f5567201a7f14964ce5b8db49fc1bdf449f68b5a6c219 |
| vector_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-e3f0465d42c3740f092171a80858b7d4dfb88420b09deaaeae251218326b5640 |
| graph_prior_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-974d7623ebd5eb175e0a3d74823c4e8aff2930d4677d89177ce9bfc940beaf05 |
| learned_route | 3 | 3 | 4/4 | 2 | 2 | 3 | 2 | 0 | sha256-d75ca0361714b9a4396b02ee0a5da369d6344789571fcebd343aeea39c9b90a8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | deeper-story-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | deeper-story-turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | deeper-story-turn-3 | 0 | no | 0/2 | no | no | none | none |
| vector_only | deeper-story-turn-1 | 100 | yes | 1/1 | no | no | pack-5ac3de9a | sha256-dfb9dc2a53f4ed8d5675642018b1624c04a96cc875020d1b0c482b6e99c8777d |
| vector_only | deeper-story-turn-2 | 100 | yes | 1/1 | no | no | pack-5ac3de9a | sha256-7db95358e737f39356851a296581129af9e8d0bd21ed0cb5247d8404e1d2ea31 |
| vector_only | deeper-story-turn-3 | 40 | yes | 0/2 | no | no | pack-5ac3de9a | sha256-d9a32542f181ad94cd3520bc6fc76b0d66ce030bb7c5e69d464f2db7a3f13c28 |
| graph_prior_only | deeper-story-turn-1 | 100 | yes | 1/1 | no | no | pack-5ac3de9a | sha256-d9a32542f181ad94cd3520bc6fc76b0d66ce030bb7c5e69d464f2db7a3f13c28 |
| graph_prior_only | deeper-story-turn-2 | 100 | yes | 1/1 | no | no | pack-5ac3de9a | sha256-da058e93225ac502c9b3077e6bd7cef3475449dfc451358c4f02e96bfe185840 |
| graph_prior_only | deeper-story-turn-3 | 40 | yes | 0/2 | no | no | pack-5ac3de9a | sha256-d9a32542f181ad94cd3520bc6fc76b0d66ce030bb7c5e69d464f2db7a3f13c28 |
| learned_route | deeper-story-turn-1 | 100 | yes | 1/1 | no | yes | pack-5ac3de9a | sha256-dfb9dc2a53f4ed8d5675642018b1624c04a96cc875020d1b0c482b6e99c8777d |
| learned_route | deeper-story-turn-2 | 100 | yes | 1/1 | yes | yes | pack-57f6dff0 | sha256-788e7aa930b15747d3fc8d2d8cabf40f87b728aa6fd05aa44e1781f9fe048d76 |
| learned_route | deeper-story-turn-3 | 100 | yes | 2/2 | yes | no | pack-9e4984cb | sha256-f969ef8dcb45686edc79a50109c0a657b6591b1dd1f1f11ff35ffc5478e7f52e |
