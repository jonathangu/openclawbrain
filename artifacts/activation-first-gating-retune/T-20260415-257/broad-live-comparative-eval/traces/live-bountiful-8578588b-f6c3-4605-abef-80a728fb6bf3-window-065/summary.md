# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70f13fb97860e498bff43fd911f822bc4b32beeedb751cba59f5d4dbbc222fee`
- fixture hash: `sha256-4b83c79a0fa7985fa56fd30e5adc48c17c08400f23f7f1fb6e39ca28d5589c23`
- score hash: `sha256-3761f78b6534c901a74bc82edfd915fdf7b3564082106ca33ad33312be3f92c9`
- bundle hash: `sha256-48cf7e8f2b19566aa2c4b2b2ff2dfdf8c5135001087fcaae3a2e4e7b358b8f5e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0c0fe80d5815140a054bf2e5d9f28b58cfd48348e622475310632a273c4a648a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-52d6a8a60cd7e1cf3ef8962006dca60fa183c00c72e8a6b35a52c7f3590b3979 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46b98fc88ab42f232f16312546f470dc08d2a72a0ecca1246058bc6db5e4df82 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-0a45e5a8baab835bf489e21ee6b6c54479b1ff1cd2119bcd838002c88c9ef8f4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-66905149 | sha256-70e4fd61655a930df5d827267b456f57f8bc77fb8760efa5582e01441f765046 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-66905149 | sha256-1f90e49d41157b126f6bcd511563e9daa55f3072255591ca803d9f969d5bd93f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d7f81cf4 | sha256-829a727714db6792622c4d63a5c0de3b0f8dd3e4bf7140b0685361074402acdd |
