# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7206817fbe9864fa741e2aac4263783623734861273e6c92294a7e71e4bda31f`
- fixture hash: `sha256-3fcf85ac262f6dca9a6b48603643e7ca5bbe3663229b7fc7238b9b7fb3303591`
- score hash: `sha256-9d7eed78d9b62910a9cf18529b02bab47df025d0ea8e0331a6a3e6e4a5c4e98e`
- bundle hash: `sha256-675ca5eeb68a2e3a0a2ec02b2bfb36d420a5aa9c588fb428808062e1f2d96720`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a6acf0ea4807b1384f37283996c98fb6c5d3cf32e52bbe94b1a201a85fdc539 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-254b21b9fe719e496886ab6ac68ebe0af30375e36713ad86e4745f39f3ad8722 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5f7cb2cff570604cdf046e4237d336f46dc6fe2262abe4b0cade1326398fca43 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-2f048d27aa9aef1f20361f65901263c5f6ac66ff1e410bc8bce5799d8f59a095 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6bfa9a16 | sha256-bc4ee7a9b2666d4cdc5e1237f0a657e69f9b6742c42d7ee507883e8fb4cc9b81 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6bfa9a16 | sha256-b5caf7e8d563d38c1a30efeb8ef15e52da66756270a3ada93742166cff586824 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-908fa48f | sha256-7c6fdd99efe88cfa5638877296ecca7c9e461451de21f01380a92fce78ad51e2 |
