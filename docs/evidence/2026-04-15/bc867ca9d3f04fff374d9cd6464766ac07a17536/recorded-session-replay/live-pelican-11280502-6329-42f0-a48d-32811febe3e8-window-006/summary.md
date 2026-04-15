# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d39e73987cbab37d8621de930c333f3c7648788fff43ee43b47bd6d1ef64fb83`
- fixture hash: `sha256-a074cddd6e9248ce93aa0306421d446e0244240e7cd7f2087a7d75eb50352127`
- score hash: `sha256-840659bc276048a1e25eb7a9db4988849eb03197c99f375cccfc3ed9e0c3512d`
- bundle hash: `sha256-e585a6159d0ef366052c07832b761986e4991cc483e8ca54da63a5fcc7772592`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2f53febf0867c1eaf1f16249136eed66aac774d903a17b5eedd4459dd80af44d |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-6eb74329ea4fc80b4d5539ff965e7ecf1296725f3abd73103eaeeb4f8e07fbda |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-9780d5908d5332e4993a076ebcc885c7cd931f4c6fb6d899a136a088da97dd29 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-b3f2bc849fcf84d0d8c4b35658e30b85f14ac6efca89cd5d5be6250044c4651d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3c1c8590 | sha256-7700311278171c21290a687cab4a18ed4fb7fd5bceedc6642a0b778e250fd4aa |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3c1c8590 | sha256-8fc94fdb9306b99cbefdffba1299d1eda2eefe0ddf2cacc5c17c30abafe99ef5 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-3c1c8590 | sha256-7700311278171c21290a687cab4a18ed4fb7fd5bceedc6642a0b778e250fd4aa |
