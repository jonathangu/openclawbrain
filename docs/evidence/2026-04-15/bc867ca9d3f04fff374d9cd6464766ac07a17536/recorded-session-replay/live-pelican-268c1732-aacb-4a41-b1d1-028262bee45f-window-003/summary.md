# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-01a3188870712e041c94dac038c1913b6a8275f11c9b961a30d44d4a9193a2ad`
- fixture hash: `sha256-61072aa6754828e3628c89803b9d747baa926b5fb67bd06b8dcc6e5a7d888974`
- score hash: `sha256-dc90becdfcfd757cc45c9c356c41a6021b1d7d35d4130568f7a1dfe730334bac`
- bundle hash: `sha256-954c14e0350b2915fdff08676711b1dbb1fa6ec152afbe7687364d6dacb254af`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2838eac92b5037b9e7a88f6a187516f6cafc0d1eb9fd70438eb0e0126665d9b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-49460201061d980933c91d5fb4a0f735fdeeb839b19eb7018ee101b1ac79f567 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e74f465064fc1f697d7398dab1146f6a17da9bc9dd7c29849f5c2ddff102232c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-4586f96f8d179f95e07af5d3085ea5e9dc2d82ff37145f582788c7bec8f908ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce6f5c2e | sha256-4baaab4f5b8137770bdf4e8582c34701a0c2f50c62d266bd2db3751f208b01b3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce6f5c2e | sha256-8dfb4916cdd5c32139ee2724930d6e53a4bb30ec6c686104c2ab5b2c94ec0086 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ce6f5c2e | sha256-4baaab4f5b8137770bdf4e8582c34701a0c2f50c62d266bd2db3751f208b01b3 |
