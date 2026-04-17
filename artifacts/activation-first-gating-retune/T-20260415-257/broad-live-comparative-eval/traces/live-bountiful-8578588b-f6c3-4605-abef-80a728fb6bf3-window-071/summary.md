# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3081fc6cd2140702648472a4581d5d967423fa00a434b9483a180d3d9c6b8fc1`
- fixture hash: `sha256-b5ac8e7827ed4f12c1ba7efd325a7c14f155c3ac4043229edd168af627bd6e56`
- score hash: `sha256-3faccddaee2a54cbf1ec7a400e4c43f2a7fbbe824f6ee63ec97ec1057de8be9a`
- bundle hash: `sha256-5abafb6e57a18cc53041f5cf2762fe85f7bf1c8d5af4e38037a9fccda4fa9460`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7d107ec56aa4c4574e497f7d1f6692d0d1c30fec9266f7ec344bc3b377fad16 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-46813b69793a246a404a0a048708a53bea1065400e54f69f4fedfb9b311f60fd |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7c558c72130922bd3a07c807bd3c6c5b2a3a673fd65efa1f4d7cc57f6219d17d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-0e1d1a934bc984e0210b2890db1b88eefe9536e38e75fd871fc061497683bd2f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-434cab5f | sha256-10b0be16f1df821f15466ffdc5f432326dddd9654ff8ebf708e2335636ffe10e |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-434cab5f | sha256-10b0be16f1df821f15466ffdc5f432326dddd9654ff8ebf708e2335636ffe10e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fbdf5392 | sha256-df2a1007be8eb0f5bdd02c1304914c70a532067e783b9758632ab551a41d357c |
