# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bed00a4e07440963fcb335df85735aa4cdf299e8eeeee26d6072510f9d967592`
- fixture hash: `sha256-7b9b3de84eea4b8f6489862313ae3b9c5b0de1ba49de793c86ffbf0e24eac4d6`
- score hash: `sha256-04b6d0731fd8b1820985b797976fed8c6dfe6d43c1bd05fe4a819460e0d9a5fa`
- bundle hash: `sha256-8a538e3da8123dbba473bc4d1fc3bc9634ce04e7a0dcca68e027e8754a328b5e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7244838abdb700b7974c4fb03ae1270d3910510dcc5c175db289b9a82a5df872 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5e8c8f6390db2f3f9f065df49b3c191e16f262248fd7c7f278a84a863832bf5a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cd194ee4e5310cb88b2fdc9b3c82186f872251cc6adf1350770cf2933ea625b9 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-75869a8893de22e2fddb69e9c2d12bd149008544060ef9f049be99d569449134 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1d5d58d7 | sha256-5b444900bd9a85fcf4c2ae1f9e1a378a24381c756afee27ebd66e3f93b5caa05 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1d5d58d7 | sha256-5730948e94d1f336a21f641fc32ac4138dd9de2a984193d1a4602ed77b9065c2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1d5d58d7 | sha256-5b444900bd9a85fcf4c2ae1f9e1a378a24381c756afee27ebd66e3f93b5caa05 |
