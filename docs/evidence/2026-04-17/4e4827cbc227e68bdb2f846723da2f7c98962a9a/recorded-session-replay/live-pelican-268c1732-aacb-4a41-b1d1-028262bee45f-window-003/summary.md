# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-01a3188870712e041c94dac038c1913b6a8275f11c9b961a30d44d4a9193a2ad`
- fixture hash: `sha256-61072aa6754828e3628c89803b9d747baa926b5fb67bd06b8dcc6e5a7d888974`
- score hash: `sha256-b34138c4c30ddab0bfa402e3b3ea4cf7dd4a8ca8dc7b1d05b385af024de399df`
- bundle hash: `sha256-d8fc4015624e7c0d60bd8a1dbb33fd81ede55d521d242ea9581b36c6ea01f7d5`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2838eac92b5037b9e7a88f6a187516f6cafc0d1eb9fd70438eb0e0126665d9b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-812eb5d7d26e56c918d3508db7a73d8d64c9c74559533b15ac95d3b63f1eed0c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3430f5c83d7892f74cd940cb32815b0eb9f5cfc2222a81624f3e685ade2375b1 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-a417e9cffbe67ed1401ab1e15ca98f95843a97cf0135ea3da21dc0881e6c1a1d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-13aae70d | sha256-4686e85e2c3079ac2a1001130d42febbb4b2dbebce74d28a4c22b73d460bd928 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-13aae70d | sha256-b8db5ee7c68834af364c2168be43c1fa45d1e25acbb6df7063e0408f5a480eec |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-118ee706 | sha256-2d195e121a3432fe0f0dd2120d81dd6bc01d0211b1f9ae48fcbad69745957866 |
