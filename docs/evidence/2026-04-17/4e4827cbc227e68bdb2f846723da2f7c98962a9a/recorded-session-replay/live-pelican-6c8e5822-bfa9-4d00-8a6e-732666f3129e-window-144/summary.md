# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0149e3caf93b3375cf02c24f74af73ff26b7bc10ea672fea0331d56ac334a82f`
- fixture hash: `sha256-8eef5aa851168050667187c6a1f16965243d4107da455697233fb94b6cd8be15`
- score hash: `sha256-41736b837fcaa61367f747bd3b1abc5dee9746f6c628930d681d1526707ff401`
- bundle hash: `sha256-e758e9957e576c2868eab757f2e9eb0dd94d4e22c96d9b5a94389ed5f2b10514`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-beac0c48f82ed7e8a11f136719a9c12038db11daf2070f49f0ee8d4c618e927a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb77ae4729b056ae6b701b49423cc8fe95924e406cd092cfc96dd7288e721571 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-837e2b23670d50a8c9ada3bedff29aeeee3e4c729a24e63b537d499437137676 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-df1d84f4b9475d1e0ffb9f648c8c254019c52f2937d2a28f008ed241b673d0eb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-72a3569a | sha256-6304b5a58f77398426febc9414e671e7d61c08ec2c7ebcbfb371bec99259e7ba |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-72a3569a | sha256-4704d8190103b2384678d7d799c21bbe3d6ebb8a66c627a060d4cd30984b79d5 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-acc5ba27 | sha256-4045bdf3260ab2448d36f527c7f96644f26e8e7c93d27329259b68a260d4ad94 |
