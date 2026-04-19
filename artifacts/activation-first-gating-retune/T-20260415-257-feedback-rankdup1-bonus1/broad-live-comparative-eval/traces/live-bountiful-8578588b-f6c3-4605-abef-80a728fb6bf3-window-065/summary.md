# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70f13fb97860e498bff43fd911f822bc4b32beeedb751cba59f5d4dbbc222fee`
- fixture hash: `sha256-4b83c79a0fa7985fa56fd30e5adc48c17c08400f23f7f1fb6e39ca28d5589c23`
- score hash: `sha256-92235445d0d8f1b16fd526a222bd72c65c4277cb5a7d7003f4308bf92b2ef2f4`
- bundle hash: `sha256-39f14f8bb184a1634208be387c0c718863a9c1670647bb669c101216391a9529`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0c0fe80d5815140a054bf2e5d9f28b58cfd48348e622475310632a273c4a648a |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-010304fb11c372fcd3bc25cdf8fcb13587f2b48cddab6d19687a965c18afd3fe |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-700f61a603d3fe7b8103bf1d4451e317c1c6672dafc7e7029e06ab4047f2f045 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-0da6085a6232260678e5039fe409edd7f0c9b296c440a424f3b0a71edad81ddf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-d7f81cf4 | sha256-f42f51c5e640adff7a471c37730422704b7966477a2a6b8c95225e419229e122 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-d7f81cf4 | sha256-4f8889a624aec01a3986eff29ddfe5c01f15af8edc671b214adc3d1b4f1c4ddb |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d7f81cf4 | sha256-f42f51c5e640adff7a471c37730422704b7966477a2a6b8c95225e419229e122 |
