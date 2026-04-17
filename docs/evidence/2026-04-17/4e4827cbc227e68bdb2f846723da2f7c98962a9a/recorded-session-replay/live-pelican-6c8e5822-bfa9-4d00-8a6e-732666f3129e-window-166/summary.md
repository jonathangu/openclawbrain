# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ec7f0ca39d2ce8e4aa075852c984c14df45efbf7ebb099adc4d8318c646741f9`
- fixture hash: `sha256-1eeb0e4e14f003831776523471001891e5f51483edf8cd0fe82b3b2a7a4e72c2`
- score hash: `sha256-9434969638bee4e74bccb3c8550d064017a852d15ef0d7483b92cfbb03e93d84`
- bundle hash: `sha256-b66c8bbcd67753ae641157c3d35508db3bb2488ce9ed658040e67e1860a5a524`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d39232e9e4182be91b475d1dc774e142ceab1f9213fd98395428e4f29aee341f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d39f68f37ab8b4f6b2a6870d1e477d328347f24899f5fa1823950d43d75f56e0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-76d0c2f69823f3833e7886a2faa46603c94041f60743fcb8d74fd918dc9cf0cf |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f882d63b07a5857ed3a19adf20580ae0f1bb38c489122a2e252d7c560983a04b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-feb8b5ff | sha256-790ef1f5d52cf10a4552ffbfa26c1b42aaae60096af20fc9005b699059aac918 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-feb8b5ff | sha256-18d955fd5ba02ba3713a4d7c4f2d575523bb084b986aafb8ff78c4871c9cb930 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a1b39f8c | sha256-69e510ab0b6ca2b692d5bac7aa8229ae774185281ea841728625c0e8b7805c77 |
