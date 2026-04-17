# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c32e2ebb8b7b9b5fb8d50fc85f34ad187e87c05bcad5baa201a1c538d7a405ab`
- fixture hash: `sha256-e23c11c000ae3f195bff5e2ea98696c33b399e18fdc28541e3c50f4b667d3e58`
- score hash: `sha256-dd2d35946ff5ee0482bfbc47e27f8080df5d34246786d861a6f8a0291da1df21`
- bundle hash: `sha256-a55a27bbc260d22fc7b243bb753b4b5d4bf8fe172b9ac47c14ed1a7278900141`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3b40ea9a22b9a22398f998c23b04d742ab7923c5900fe44bcea6dd68bb464780 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-601b7a3c16c1c9ef349e934793d9cdc46ae9808f310daea5c47d0f1cccea06c2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-40724cac189c3d3d3aae8e6291432f60a4e30cc7169aefe74f54a8f7ea6cf952 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9920f460b497db3df7ab69f476bec8bf8ae2b51805d46792e71aa831716cefb5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-92916574 | sha256-59916202756e3967c53c4bc2d2a0de0419838132659afc57e187e1d613e10e21 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-92916574 | sha256-3de983a3735a5bc7b2e5feef464455c4b5ef1d58d940bacb282edca4dfafafdb |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-61b30f2f | sha256-4a865a5ce54c33d277e081c8892912de329dd8dfa650ddd695ad84fdc8cde202 |
