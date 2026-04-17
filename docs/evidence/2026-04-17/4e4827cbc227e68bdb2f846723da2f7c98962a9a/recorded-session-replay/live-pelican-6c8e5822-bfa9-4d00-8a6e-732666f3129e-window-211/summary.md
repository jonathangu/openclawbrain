# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8f3ad7fd7e03d5e6a620d917f9922d423fcf350f433bf42dd24d49c0d04613c`
- fixture hash: `sha256-d090bc75588ff2d651484afffd5d21c674237c8a0eae19ac1a18854f75e95a21`
- score hash: `sha256-1f5bc613ab6c244f83dbf065570b5a6ea612734537b902df8b3b36ce2cb65092`
- bundle hash: `sha256-fe872dc36dc679ecd45ac039aae54dd2db4e7c3195d9a0b0eb591c5013657a5f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a44aad7aa94fbaca9460011cc6ae9061f9cd3a6c6afa137136f8bba1929488be |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-5411520e9dedf82de31415c98e9d229c2b4777c84ff0c16e4017fa2b3de3678a |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ac1657ee613b856da1299384a5b4bd5b9c7c51f84c0ef99b242df96de67f2b78 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-5dd49cae310bb1d860ea74caa8a5982644534656b0a8b1e4653469341001934e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-46d17647 | sha256-10eca4ecb4e3207d74f11481e6bdbe7f818d937685cf74b90011fd33c139395a |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-46d17647 | sha256-a17ca58d2f1abe3ae83dea9ce26555a556966c1bfa168dd7ce695aac12f55534 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-cc81b4b0 | sha256-eac48998c2581bf5dcffd159ef63331799e3ff0aaeea4960485ffd361fc005cf |
