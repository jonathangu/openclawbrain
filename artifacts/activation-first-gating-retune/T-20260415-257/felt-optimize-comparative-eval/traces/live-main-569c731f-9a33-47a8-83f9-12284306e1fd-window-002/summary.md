# Recorded Session Replay Proof Bundle

- trace id: `live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d27db54f25bb1682fcfa202523b5f1c6efccc7e2753d8e02e54ba11f6e3abbc5`
- fixture hash: `sha256-0e0db1f3540c6bbafcaa45e48b36b0aa0cc986ef0dddf4d7e13951d4b175679f`
- score hash: `sha256-f2c30528e3c3f9b5542f97a30c317279a0397695da97ff13f9b3cab028d12d55`
- bundle hash: `sha256-6ebf2871d9e6ba47e5db63d041da20c19efa8f99105eae6a5b28f492c54a95e4`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68d5e578e4c5ae4cc03b669502125f551fdaddf1a402ff43f610e723b5147ffd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0d2187c984878e6e13f9ac093851fa1bd47ace17e250b754ee2b256ef5c9dfc |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-25a5cff5e07a7eebae89b13f5bee9bf8830ea1fad6b55995d6ed16e88460dd9a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4cb0755e | sha256-a6575a2a34e37632db4e8dae51318e576abf80df7f1d08f3d12ac40271aaae7b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4cb0755e | sha256-cb7565fdc41d87d4488e81046afba56da49ae161ea0ac57fff001cccfc774095 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8c3a954d | sha256-40e4a65d4b647eafec45f9dff92c821615f71c043c4dec21a7bfe35e69231cc3 |
