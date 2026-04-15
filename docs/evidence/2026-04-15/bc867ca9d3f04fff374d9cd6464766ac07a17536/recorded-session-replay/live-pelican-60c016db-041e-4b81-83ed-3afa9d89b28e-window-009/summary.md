# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6c49ca2fe441629d93695e938073dd41facc650cc9fd301e1fa807efab482f72`
- fixture hash: `sha256-a51ac08634e3c4803c3d3973ce1a7c858ffb1429844452de0ed6e3279b36730b`
- score hash: `sha256-23c3c1e10acf282ac6153b577e2e0be5bbf9f2567eebeacad23b6b36e7802b3f`
- bundle hash: `sha256-fe085ffc75222738fc786b8326aa5f172a2c28f06f4c8cb6675384f7be5cc1da`

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
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b97ad03bd603f2c5fe2320c7eb49a9e795311f1793e19506c59cff337e70c9ab |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-39785dc34eba6bbe649779036a550bef269e52bd0b76d47c3d55d441c739df57 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-f4c56e4ce1bee76a3fe3ca91099ffc4dd27ad43e20db781c6eb21e80ed73991e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-a07fbfff | sha256-e3b402a73b56ac0b441a0c6c7fbb221477b83bf2963d5b174cc477587f2ceedb |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-a07fbfff | sha256-b03661e80237923d78aaa216e16ba96755b6381ce672445d593dcf68f539a1fb |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-a07fbfff | sha256-e3b402a73b56ac0b441a0c6c7fbb221477b83bf2963d5b174cc477587f2ceedb |
