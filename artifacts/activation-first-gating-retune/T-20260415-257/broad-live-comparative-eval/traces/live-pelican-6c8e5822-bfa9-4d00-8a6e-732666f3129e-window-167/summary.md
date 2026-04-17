# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9808f8fc2b34b9bdd2037973a4af69235ae13bd43fa03c06a9cd2c930faaaa29`
- fixture hash: `sha256-c59bcee0d7e5004e8699b7491ca609cee1baf21baa2824b5dbd8c966b365083b`
- score hash: `sha256-c847613d72980e9ce996b177e64d4248e1ac998614d24d7a8740187e019f1f51`
- bundle hash: `sha256-8e55c60364532167bf2fb0e637bb308e5415789e9ee866d15f0bb2c1bb45e264`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68b72eac2031bede9aa8770eaa1f000f5f4b3a15976311e630a954289393b0bd |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-92dddb2deeca3892ddffaf30cbb07a9bd01f15a7ac3cf9e6aa0c5e4b792eb8dd |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-16b3c60decbb702e2463cc5a8131ca1ea27c114e4ffa510967c76443c272a358 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-bfe7b0d1ecea89761c44b9b6ba0ae5a927e08ce5c6887b9599ec19cde505234f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-7b988315 | sha256-1ad1ff4d64cbbd09c090b261dbaea516865e3003a487f15c9874f4dd76c87c24 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-7b988315 | sha256-a546a63bdf27f22afa971ff889ec430c26cdc92a5c5250b37efc2e5be373fa88 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-848d89c2 | sha256-a905e560b0398c1a79fdb512358c522acc266851ef99d4dc231d24e2d4a65bee |
