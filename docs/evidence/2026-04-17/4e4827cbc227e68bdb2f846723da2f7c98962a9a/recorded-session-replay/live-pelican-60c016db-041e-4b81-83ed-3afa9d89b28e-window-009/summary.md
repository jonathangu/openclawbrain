# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6c49ca2fe441629d93695e938073dd41facc650cc9fd301e1fa807efab482f72`
- fixture hash: `sha256-a51ac08634e3c4803c3d3973ce1a7c858ffb1429844452de0ed6e3279b36730b`
- score hash: `sha256-e5ccc3d25058f7abed6ea8dfe8b9e0b3ffe917d01cd8604ff71fdfca04091f24`
- bundle hash: `sha256-90e2c84ddc31ee64e486bd2e95f371f4d857c8f0550b128b24faacd46b595ac7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c2e3cc9fd6ee1604d2c526310b1ddca47a11a50e7f9573ba696d4001f01dac |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-84b0a0ae6b7059901c567c2ef1428386c81cec18aa996315244e0476572d8c16 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-d32847e527e550813801c2967c8d37a9b504fbbda146c873924a1eecd787c720 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-885781f7349fc94012bb46ebf63dd858ff63b7590759574dbc0df48951d70b18 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-4fe8c88e | sha256-e05d4ea2c31b86f797b3b3a494f04691a7bbc3ecf004637390b7052c2df7095a |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-4fe8c88e | sha256-40787ea724ba1502713a059d5e59b3d336ccf6178dbee2ccbf6c349f35c6b9aa |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-d3589ea7 | sha256-9438d1d0db38563eb6d38b00573c00374ef089dc380435ac0c7319f20791fdff |
