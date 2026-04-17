# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60359ed78d5b78e9d115bf8cb9e9ba270e0f90bac409bf6884d4a443b2440f94`
- fixture hash: `sha256-0da91a494c8a34b6c27eb293958b781dbe6bc334337372f9fbd368fd3d0ee08d`
- score hash: `sha256-ff4f3caf725ea23ab0e427c73c2723a937dea1514b101825d40e68f1abcf2bd4`
- bundle hash: `sha256-678da907ec6b66214db9dc499d1da3f506fff1298b215fec1e6be159301e7884`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c77331726a9326f500ec3f7c3dbbaeae387d368e17255232ecaec7597f897fed |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c355d6a9ec0ee56165e193f083841874f70cfeadd33d22a0da2c52155828d074 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-10af454f8048630ec101210c60994e4b78f28bb27873eef17c1b5ca235c071da |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8f95c9ddc71337ed02572e523bff126c969c869effbc2f64b903cef3c4f50445 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f4701431 | sha256-bed396ee9e739453450cfae997c4e33dad56ff1ba03a7d1f53be3ace7c2513ab |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f4701431 | sha256-abfc5e736f378d25e2df9671dbd6b9e9c3073f5ce899663f95841286b02a23d9 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0946344a | sha256-a88e6041f8aae7742ab7bd339fb3e0a3a06f26acfe6444ca74f52270172fc05a |
