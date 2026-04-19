# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c58ed04d44aeb04071688c4a26c4c689e25ea007697f349c3e4c8fcbe3bda533`
- fixture hash: `sha256-ad70501e856aff4a57d924d7225c4dc64463e70da2f3e42777305ef85fb46a26`
- score hash: `sha256-530e060d719b745c3570b6e2285c984f7b8dcea2728f6913b2e5be17c7d8a637`
- bundle hash: `sha256-65e7ddfc05b163638fab2b864732b8c9b00b10450e074ce7fa23c7893b6f6ecb`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d770aca06ab90e2e0a0ead714079ce642ffbbb18580e6acfdf4fde922a74f5a7 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-1500b35820290b861e23989d99efc3dca79ed9ec805eaf62234819d9e63736e4 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-d2a056541930d212e6da3e0ed44f68fcf58304c3218d85433dd32f54b7a382a3 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f8652367c584270474ed06d9e2dbd9e78873e3af61fd4cf55a1bf4fd03a3d237 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-9373c9b2 | sha256-e7b687e4099693a1e62046ffc6ce1cb8cb8f178024db31fb590a19247fb5748d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-9373c9b2 | sha256-3ff79e3b0141887132c6c4b7382d4fe19b257c1b22c2dbc6d72980f72ce1168b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-9373c9b2 | sha256-e7b687e4099693a1e62046ffc6ce1cb8cb8f178024db31fb590a19247fb5748d |
