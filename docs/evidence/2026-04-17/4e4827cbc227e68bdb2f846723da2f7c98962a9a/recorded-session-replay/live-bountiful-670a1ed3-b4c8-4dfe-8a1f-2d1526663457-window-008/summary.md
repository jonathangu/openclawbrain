# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f10c00dc4efb180b5273900a9e561d1c614344a77050359aaa2d54aa27cc20d2`
- fixture hash: `sha256-43065829df1e95ca79dff07d99e5773679b5561b6bbdd3945d317201ab2cca51`
- score hash: `sha256-4770fcf469084c4ff48f3b290050a20f2599f0ab88219d5edf20e4806b355068`
- bundle hash: `sha256-e686632ad86bc25c2c9b1047cf5379b4d98ee4e7f78766b763462dcbe984fef6`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f1460fd13a644dccb389d5e4bb97bb20a28fa61d221da193a36a1bd2b7379c0d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-26fbcf7bccb047baa2c4c8b7a5c4bfbb9bd3fbb7cf7a6dae2e8dd4dd2d99d754 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-17f444a45a53b3f91f17b7fdddba39429f3273a6de404bae82dc4ed8b4a588a2 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-8dec44198c09a70bfbf8963f391a1e458f0bc55e982e2722fa12566a7f28ae28 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8d37c96d | sha256-5e14f242e9e96a68d95eff33db9afbed2bfb42816625857aaa6d8a4763a58f3b |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8d37c96d | sha256-0022fa5bb434cbb05b1568869e8420809a46be4825da231abe7801fdb4b79515 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-49775b22 | sha256-35f8fe8435176009eb50862a107957642485ae99af0d316df1e8c8e0578b18d0 |
