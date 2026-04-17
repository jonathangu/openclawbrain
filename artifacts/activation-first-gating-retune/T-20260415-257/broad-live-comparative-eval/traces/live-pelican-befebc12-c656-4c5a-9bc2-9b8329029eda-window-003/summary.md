# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b0cf5e7b3f41bdf7d892185413608401c24e9c3ad252c16335ba4fe2f91cdd3`
- fixture hash: `sha256-ed8248c9b476e9fb2d02b9891cc8e11da35a8ba49c308ca9793fd2e0cd5daeaa`
- score hash: `sha256-76f6117753e96fa9aee5f9503e8c40b7b9bff3df0e19c65b4bb2cbb0fa881435`
- bundle hash: `sha256-8ba438b7eb5a1b5f48b6b18daa77d3cd5855891bf0429f7882cc49511c119169`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-368ba7ad0e0062707beb6bc226c2cae8531ed592ec4225d05a99c6ab4df81531 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6c3a5eb68894d7922c477223c1978566398c38dd26706c64a05ef73bf0dc7246 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d82b2e287dce4d04b4e8edef52f59ee9b73ef5ce3d6e3c180a2d9253aa8ca161 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-4d4d9737dd4cef944eb58f3099605bcfc336e4c7cb55d1a3b64536aa91705538 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f1c3c4ec | sha256-c9bc888714110f7c7cac78509e245376457bf5a6ef193a1b2b6586162bedb408 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f1c3c4ec | sha256-c31b41f7565d38bb2a8dee33e81c357efbfee4bed492857b969978a520fe99b1 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-6f7f3a11 | sha256-1042340f73960a83a6f65bf67cb40507da18e9594079a9dcd54d4d9d1b3722a3 |
