# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d380c2fd8773059a5893ffff2d380e86ca0f972a4140732a5832ea7865e5c2a`
- fixture hash: `sha256-55f386f545922fe7856a581e64b7fa651b1de1ce7956a55af05d3b2bdc86946b`
- score hash: `sha256-74ef290b0f387616704f2d60b214a054a5427283c4d65b60af24e31133e8c58d`
- bundle hash: `sha256-f181ae83fc0a21e995ad7813476e75ec54679ee1630204dd23718552e2d6d4ab`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-63284d4d64db3291399ac8e17a28a524a22240af2c68e8497ef443766b42c4ca |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-033186f2a060f0496f95ee0afce08c47407121db609eb79220cd8c267535b3cd |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a513a7bfd6db126aa754068b7e23c7d69afbaea3ed32dbc2d78dc0019588d045 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-fde117117896ef5ed3c46cefd470ffc85ae682832e970f23245d7ff2a24e7d61 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-22c21243 | sha256-77861a4cce05350e430c8fc48f726c6c55bcaa5d329289a6aed9183e106ae73f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-22c21243 | sha256-72618248277f37913bc96a274e59e0c5b9fbbfb4c02004cf805934768bfe28ce |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-3ed56cbe | sha256-3b654f74ab51806dd6f8fc1de73e1f6bb1a6bc1e710f07bfa0e27924fe0a5335 |
