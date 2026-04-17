# Recorded Session Replay Proof Bundle

- trace id: `live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d27db54f25bb1682fcfa202523b5f1c6efccc7e2753d8e02e54ba11f6e3abbc5`
- fixture hash: `sha256-0e0db1f3540c6bbafcaa45e48b36b0aa0cc986ef0dddf4d7e13951d4b175679f`
- score hash: `sha256-63f8e44381a38ff1f7e4e3cdc71b7773d3fab6a6a25f5c723cc1af7273ad68e2`
- bundle hash: `sha256-67f45bf80d746e8a1ce2f165c293c725aca3dc510fa0d2a5d02c6359512fdd0e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-71d12c78bbd92c17749c2ba921bc24d7594735564898b2d4c08d5a5f8badb93b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-615d8945a2067f44fe069b5e0ae68fd46713a4323d9281ac6a68407296ab9a52 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1bfd2bf56fc96c586c5065ecfd604b14e873cc951d08f5133de7fc9ac44b8f2d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1860bf586e509a310f150a005df0ae3fdf0b4e5f6182ee2642b4f6e7102ecf58 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8f5b1cb4 | sha256-25269039e95450d6cae39d8be7d656a2578250492eeca909e06f9d4f91002ea4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8f5b1cb4 | sha256-2b56c20aebd5791e8871302aa28f05684484f777fa0df3ea2c3dc867986f41a5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cee53ca3 | sha256-1ef9f7769d225404944f587cbc13a6311e4e3095548174fd01d98463c8e54442 |
