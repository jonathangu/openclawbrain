# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7206817fbe9864fa741e2aac4263783623734861273e6c92294a7e71e4bda31f`
- fixture hash: `sha256-3fcf85ac262f6dca9a6b48603643e7ca5bbe3663229b7fc7238b9b7fb3303591`
- score hash: `sha256-71f828e68841574a5c5ac5cd1b39f6672369ee33bb14288c2c46885460ef0880`
- bundle hash: `sha256-7e03deabf8cce532bfe473ce79635217400ec31123db9fecfb53f1c7fe05353d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a6acf0ea4807b1384f37283996c98fb6c5d3cf32e52bbe94b1a201a85fdc539 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-555a2ce29f2fe575c63004b601ff0f4ec2a6b5d600ff5a8b5f019982f6f360c6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a2f2524fc20a15f6228682083ba01c0fc6ab782e9cdc59e52c5222d49cb37c07 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-321cac1dc90219d2740caabc933c852a5c15f0169c1e01a71c789ec194098420 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6ea78158 | sha256-2a95da71b203c15ffa01c13804afb47bae394bb4c4b4839c248cf646d2f94a74 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6ea78158 | sha256-feb37c16b23d5fd94d6e17d36dbcd4afafd169db1bf05af9d56c9bf57e95a335 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-933c8bd1 | sha256-78a899353a5dc3bf681916bfe4d195a6957cc41e93c58c0fae62a3e6172aa5d7 |
