# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-07e10f6820bf810e1999011ea58316d9d53ec99aa0ac7473d30a2c9a79d153ae`
- fixture hash: `sha256-54c4d68b5e528e2dc7ad50c599fd75e1b659a972d8d4c97376e292a3ef62dcc8`
- score hash: `sha256-610577fa10a342d0e0087c4bc2d67ed3ac8d96d5cc5ccaa77ba3bf2c0231f2e6`
- bundle hash: `sha256-a2952dcbb569b33bd9fdbdca3360a1b65cdbd94c28ba9ef8e67684c689f3036a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-430dcaca40205cc8d42bfba95521d8acee2a6e6c074b542cd0b9a2d9f1547939 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-079d2e1e2f2ec122554a00a8fc998f8b984593e786b767ae6754f3ae5ca5e07b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-836007dc9cef9ef7c4b14879f560cc44227dafaf9fd34e8d87df5f89d7b22cfb |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d0789d52688045add0d5ef5e6450dbf3133dcabf5c88042ef705ccc497d1d5c6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c6b23839 | sha256-0f2069da7ec51fcbd1d16da29e881300400f71844db9895974e3795c2f37331f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c6b23839 | sha256-e013882967fe8277a709900d808d8043f9b0cc051be9e5b3601606c40489c5ea |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c6b23839 | sha256-0f2069da7ec51fcbd1d16da29e881300400f71844db9895974e3795c2f37331f |
