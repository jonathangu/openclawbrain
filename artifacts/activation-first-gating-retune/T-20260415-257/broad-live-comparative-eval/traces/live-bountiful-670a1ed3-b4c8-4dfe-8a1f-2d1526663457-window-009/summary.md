# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c58ed04d44aeb04071688c4a26c4c689e25ea007697f349c3e4c8fcbe3bda533`
- fixture hash: `sha256-ad70501e856aff4a57d924d7225c4dc64463e70da2f3e42777305ef85fb46a26`
- score hash: `sha256-6c95ac812466d9050345228992c2ec3010a03bf440f7a6c9c4126179858156f4`
- bundle hash: `sha256-1bc54079c84d6d2d9c167dfc9c26eb90d30e34d32bb279784d35c24232cc7910`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d770aca06ab90e2e0a0ead714079ce642ffbbb18580e6acfdf4fde922a74f5a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2d18649d97048ec2487551edeb8e5a0ee4db739ab19430493fc5bc13e9b3ba03 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b2ad8ed75cc8c91b54b5b274173c44b39d8cd81b65be1e333bb9f8b904905a28 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-24eacaaaa385d34917fb7c3239f2c8d1076be0b312eb06bac53d0bc3d3328a31 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f7d047b4 | sha256-48bca1a2f4526b3b627d39874ab9b91f882accda3705bc031c1836c524cbc2bb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f7d047b4 | sha256-8317582824ca4913e378e14c27ab93272fe085c221d5f6b25add0aaf7911c812 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-88da4b91 | sha256-9d8286e6cfc385ff75c12c999b70a98244d11b29577da33ec5504bdf6f0f37cf |
