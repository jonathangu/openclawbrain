# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af95153d1f0a3be68251ed9ca1c6eec687f3524276f083ded8a5b5ed5deb8173`
- fixture hash: `sha256-e08d20ecb487c4dc497560c31f0ea6c918c59692d8f02eb17f1047383fc56246`
- score hash: `sha256-ae1429b1e93734cd199f7ae02878b9d63acfe8158d6cefb28a9c2326572ad533`
- bundle hash: `sha256-d0198a0946c9ddfaf19755b7fa2b73c905fbcfe1ca3767f70e6dd6455d503d3f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af28d7bfd3a7d04b34389f31647ac0f041f3d52e501bb74630c37fbf1936f421 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8e1240f9ff9d0421fe4b8f0d7dd07bac4fac6af96e571536ca400878f340d96b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-675f3090230e0f794af27a2a76395cb73dd58e7b3b1bc3641f892f7be9b251ce |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f9bfc71907c9f91c1beb8edea8ecbdeaade5628a09ce32419de28309468be708 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c9c1303e | sha256-a8265c7760dd7cb557397457dd4b92ef99ee56bb877306dea37628f7fa09976c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c9c1303e | sha256-35feb732c36f6b148bb0c363ef32147040b1e96e2292dc58df8b9a2c0f9060eb |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b928e0b9 | sha256-b790fca5718e8a0f34c5070eb0bb4a6469171ed4749cebb3075829a1d0764afe |
