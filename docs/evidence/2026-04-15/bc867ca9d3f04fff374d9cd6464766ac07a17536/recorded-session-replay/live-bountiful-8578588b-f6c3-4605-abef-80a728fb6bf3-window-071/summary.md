# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3081fc6cd2140702648472a4581d5d967423fa00a434b9483a180d3d9c6b8fc1`
- fixture hash: `sha256-b5ac8e7827ed4f12c1ba7efd325a7c14f155c3ac4043229edd168af627bd6e56`
- score hash: `sha256-82a26127d459d21ca2ea4e5def4d40f3ddaec81e22dbf48ceb3e09a3d78158c5`
- bundle hash: `sha256-dab42176a148be2246f16a5d87d7382dddde90c00ffe4603f39eb8904c241b6f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7d107ec56aa4c4574e497f7d1f6692d0d1c30fec9266f7ec344bc3b377fad16 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-46813b69793a246a404a0a048708a53bea1065400e54f69f4fedfb9b311f60fd |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7c558c72130922bd3a07c807bd3c6c5b2a3a673fd65efa1f4d7cc57f6219d17d |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-7f05db927e3746f3ae84a19a0e0b717cd31b7b1044df1dbb7c42efa749b6accf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-434cab5f | sha256-10b0be16f1df821f15466ffdc5f432326dddd9654ff8ebf708e2335636ffe10e |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-434cab5f | sha256-10b0be16f1df821f15466ffdc5f432326dddd9654ff8ebf708e2335636ffe10e |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-434cab5f | sha256-10b0be16f1df821f15466ffdc5f432326dddd9654ff8ebf708e2335636ffe10e |
