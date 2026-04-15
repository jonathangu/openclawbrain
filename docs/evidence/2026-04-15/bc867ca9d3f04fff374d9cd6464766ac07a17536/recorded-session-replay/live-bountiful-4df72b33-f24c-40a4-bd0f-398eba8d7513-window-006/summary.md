# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1c2fee7fd4eb0c2720a3ba15050df8108cb036feca9a01fcd35c4b07aae7a9f5`
- fixture hash: `sha256-61f4419f55eaa7d0c0ca68a6f768711b70a4823f4e0fe058cff8927193ee8afc`
- score hash: `sha256-4db1779b291bd88bd5be78a8664747e9bac4029dcd6c25175e8314c20a4b32ed`
- bundle hash: `sha256-0f13d36d27c1c9e73850edc4f5662d8d7cae4c244cff2ca83f75dac4a0cf8391`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3dcc36ea1001cff13b10454b28af88c47e797eba5193d74b4990d61c1caa8eeb |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-984cc0e663ff4bd0548034183d092e129059e459db60c4521057a3df82f6310c |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ad8fcaded249c7ba57f1e581d1842376035273798ae069f847af75a5b3a85f5a |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-a98e73221defbb9425ba9f875f4b95dc32b64d4e1c31f8f94993f7c188339e19 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e76e7d95 | sha256-ab9fe92bc9b6a000ed35b845c67e0af49dbf194b75ce375885f0959446e6a067 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e76e7d95 | sha256-411932577cc05b46f368ec8f7eceecda2985bee3a269c2748a18a85c17843ba8 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-e76e7d95 | sha256-ab9fe92bc9b6a000ed35b845c67e0af49dbf194b75ce375885f0959446e6a067 |
