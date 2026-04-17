# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-caf66a8ae0f50cd2a6d3d3cfd5e877c0e6e80108a88f78a99bf9d0af8e14c2ff`
- fixture hash: `sha256-d0e5149f0ea8ad48a690b98cf321460b1ba9a083ea4dc63f8728a3baa728b105`
- score hash: `sha256-8fd374cc73cf46b65b4c36ab35ac67dc04c386c28146c374c492d2b71b6cfbb3`
- bundle hash: `sha256-67757ba8e897f0ac9f2a21bf03f6492c3d10721f2955c4d49b28953a4be9e8d4`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | vector_only | 70 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9082b7d02459eb2c821fe90896dfa510257befaff2be31c5d25dfe86c62c130d |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-e8ba5db9d9f161c399e41cbd8225c8a80eaf6cc6955fc0e037c9850b22a6c807 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-890a207a4c3a04c34206ebe0e6dc056d0c9146c6757eabe390aab3395d6477d7 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-145d831174f9c2e466f25b3b96ae79bbb79496544ea3ba5aa79fd4a1d3118cae |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-c1ea0537 | sha256-65c43b51b92de7b02c96c8c7cf80948a674b85f767c2784df7ae369d89c96012 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-c1ea0537 | sha256-2d0cd69c0ca1784522d61eddd87d26c3e1fe86caf8ad441365589004f74ad08e |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-b44b5998 | sha256-85bcf302a0bfd34b435e124ec3d612c776537620350a9d7bd547960b068efa28 |
