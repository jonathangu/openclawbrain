# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adc21e40f3c3bdc2111e458183ef292b9fdba4cc9072a5e4575150e3a25e7599`
- fixture hash: `sha256-82594518eb539bcd92075469119fdd7049793972cdce0d3d047ffdabe9e539b7`
- score hash: `sha256-a7807cab98cd63170c1c1f12a92dcba24e38ec035438ce9bd46caae38b3be01b`
- bundle hash: `sha256-9375e05147cb82d8c1ac29c2df9d4b4cbefc0c9aa9fb106f99b77a6548533495`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca48aec6e03fc6ebf10d02ee2af1729bb6ff692653b0f22ac3e3b10f844865d0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dab731bd1b34ffc6fc07ca116b508db61bb85659d95aae0612f9439a72306972 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6e329aaba419ef0c3d735c00ccc593e9f2a13c33f03be46d35aba9b35694029e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d246c8211d3c7bd3995f677ba469df63afbc8a7ff56c761a333ec8e3eb8de020 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-468403a3 | sha256-5424cfffa319cc939eb927665f6846c2bc59438e13be738e18d7e52a0059e2a6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-468403a3 | sha256-7f08ed2b655697fbaf8c41a50beb73ac71812b3d79fb3c01ba1573d29a87d330 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-716c5986 | sha256-e613d82cf8f89fd3bd4411a6ba08e66f0ef01cb81b1476f5c9d35caf9a6fa007 |
