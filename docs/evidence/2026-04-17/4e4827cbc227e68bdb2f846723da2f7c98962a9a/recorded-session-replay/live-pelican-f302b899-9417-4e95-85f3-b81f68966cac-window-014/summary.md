# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a76e30e78a3c6628f4a84b691210152bcddf9b8fa0661b16388a6ca59daa23ac`
- fixture hash: `sha256-2f161d785d6fb80ca3ab0af035b3aa3abbed725f829a4bae1a60b67e83a88b19`
- score hash: `sha256-b265e0924668d9e7341d8b75d9d2cfc02f0b704c5ff63d75a9fd01f179552e74`
- bundle hash: `sha256-7602f137ff092641771584608360854f31ed001f8ae9476dc494af5646f2d0ce`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37705891572c574b5f2ed2ea56d6ec8c0372961de1b290750eeabdf9bb9948c3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9f9f9e532b9ba540b805a10ea2edeb75eeeae0a3cb272a09e401fda226e0cdc8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d41d127d18c288b40ec2bf6ca9c38343c6a41f0861702b0094d39fc9aa4936f1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-505eea44a6d00858c9218957f3eb1027afa9a7a35653727b29146d06cfc6a8fc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0e9dbaff | sha256-e4e582c3b998bc1c61900fd7bb82c9e4f0f2df5887e301acd29ce1603fdda9df |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0e9dbaff | sha256-c1b8d5f94763db5812fdbe0f4a2bbae5bfaab838d9bd139175c7a74c57ecdf44 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-5854340e | sha256-0a4f6cd36523f1819c21ea7d7c6e81d42612835edd187b0bf1a7c149c7a5ff67 |
