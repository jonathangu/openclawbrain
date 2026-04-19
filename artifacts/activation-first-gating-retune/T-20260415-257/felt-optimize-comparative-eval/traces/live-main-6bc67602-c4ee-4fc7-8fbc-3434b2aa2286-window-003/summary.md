# Recorded Session Replay Proof Bundle

- trace id: `live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b093cf106a437e24ba93bbfbea56317e62afd65cc282953b847c0fec17c90f`
- fixture hash: `sha256-f186a663337b28243cdd6e62a9c63e0bf0678cf05202237e1d19a1f17b82f110`
- score hash: `sha256-4822fc096b0e8ff443d5857d0549f15f7476f0bcfab5c6327c648344c48c5472`
- bundle hash: `sha256-10b9eefceeef3b7b0b168b22da19e86628b27d040aa92ef520106f3c689cd6eb`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea127012751163ce5c5c7b6a51409b045b05c15be13611d375e11b98fe528366 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-70e9dfe883b633b3ed144da39b7eca2614ea0261632af3fae2f048be191b1467 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-131ae0eeac14b6aff6842f6852e8bcbddfc3e34cf2bda8655ea9fe63210743c5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e87dedfe8f9f38228cade8137771a2fb68e7ce9e17ffbbcecb39d1d31b60f42f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-4fda9cf0 | sha256-c6541875c2ff8547ff90c42030f6957a5d0c6ff86a26d275ab675a6af6ceebab |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-4fda9cf0 | sha256-45e117b813da0429fb6e1e93f65eccbccc870675cc29be1873207b51307a7e0f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4fda9cf0 | sha256-528958eb5cc0db852fcc78e6a2d2db88891c50521c70ed73c161a47bab116e14 |
