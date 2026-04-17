# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e4ef5844fa79e43c9383933848095b91a4b282e9ee457dbc9f1c9f66329542dd`
- fixture hash: `sha256-7a05036812c8c043bba376d7dabd598905517ac4fafe99540ddae7c177988a91`
- score hash: `sha256-0ae8e8c26f07334d04599e1db50f0b2c17bcf922b8b65336c606e4655a39735e`
- bundle hash: `sha256-e23c225268721a3ab6caf4bd792062d9cbe04d515a7406280de39d4dab0a3b1f`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8d02dccc8f2c581b6afb2eeb0827f1dbd7b9a4cbd13481c6743fbe222b13d1cc |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0102c5e09191152a8cdc2736353a3fcb7500a1af712a2b03665e715b7dfa48f3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6deef11d685295e57270c08944f47fca49ba431c2957b6d379e03798de88a7a4 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-74c19c1c4970dcad69ea5cbc5f2c4ac4eec03fca3870658b0cc61a88d9bead44 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5a515b9f | sha256-38da3d948aa481a00be81df241dd809decbf63b7b9208add51ab71e3b7446180 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5a515b9f | sha256-024ac8f5e912528cbbfa7a5cf17836d4102d67c8fd264b44ea8ad9bb0bb861d5 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-7c5d5fba | sha256-326a0c352fcf1393bf40144ae8907641518b75b94b25c73a8c7e73b50f21b899 |
