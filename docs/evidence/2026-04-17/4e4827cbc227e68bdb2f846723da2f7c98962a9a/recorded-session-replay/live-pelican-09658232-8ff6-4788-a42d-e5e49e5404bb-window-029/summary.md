# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24701235d9bef68e6850974201809e3a73463fe7ddfd0b5cfe74a867885dc71e`
- fixture hash: `sha256-7c9db0ae094c3de40db6d4e0f20c52b15a3dee97c3144a7a4c433e3dd89b20b6`
- score hash: `sha256-13ff50478124f737b525c1452890e2d874a212284e98d50bf1169946f2985bd4`
- bundle hash: `sha256-a54ffd858ba303eba277ad4f4530d0c4e6c27a492d705132f419f2f1e91a7033`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f3c8d6c272d7556d73fb57fae65bea8046db993f5ac8290705eae6ece09a508 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3c40ff3858cc07e6a74fe1d52df0afedf0564543d624bd4cf3409269df97cb91 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fbd2991f4072fcd183bc1aa03f279d479a14fb69db4a8f11ea183f3230b3fbd3 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-6681e97f9ea8284a7432b0cb960fbbb0dc63fde7e34fedfd0a2270fecef048b7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-996b2550 | sha256-084d16b8189c1d10b79fde89eee11c27794b567d6f63f481e65f00d1216f7f11 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-996b2550 | sha256-93d31b7c3884d3df4bb4db2156524d43edd230e08d02ff9823ad9ffe94528921 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-36f1f503 | sha256-ee8646b4beb9d82d6e1ce0ea561c75e28f34fa79e7e4fd2d1f3bd1c5fe3f7a32 |
