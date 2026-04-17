# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50bf6e94091f3556fa81577b5a708e4425c4e417b6705bd10df603b7966e593f`
- fixture hash: `sha256-18f79e8bc777ea9555de29a01a8501d21c8ddf1c9ea32bbf589d49b4f4a3aaeb`
- score hash: `sha256-e0bc753175108867a79e633456e453a8f4498098b5f033d53ac29e44239be8db`
- bundle hash: `sha256-3837074199327a32ec8a1be2ed2d6c144899fc31ef64f2f4f4285932f6bf219c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-12a31c86174cd8cb94081745ed9b01e9e8efd75760d60dfa60b0b81778821ef5 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea20c8d9f0e0a2ee8e878b84f68e5f170be609c434adf801d44b0ced210ee520 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-195a5d078b6e18c8bfee51ad876a87b4edfffcf373e492fa579b41a04b5e309d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8002c4cfe3a675496bdcb99124a988be7534c9b861936aa5df35d89c8252b1ae |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7059fbe0 | sha256-c16845a9fe37aa085e3f9e57a75e97104c5a71118c30dfb880cf7bacf2d5c80e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7059fbe0 | sha256-b92eecb35551d6a115c803fad9f6d3de32b665bc5719e14cb2455ebca84a7add |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3ef4e925 | sha256-a719c584bfbbe43f7ee017da8a3066a9fc317baadda30b272ca4d098af1aaba3 |
