# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9de933020cc2b0d03a4b1b4f5bf51c7bca0c4ae8de78af7a2cf6d4b86ac284d4`
- fixture hash: `sha256-2e5835aa933cf2df6faf2714837c2953d1866a5094413604d0ec3e648b5257c4`
- score hash: `sha256-d817bbe5616ec0424a4f1a0aedaee9489293a173736923b849f08b13a4a9a100`
- bundle hash: `sha256-f33dde913e5c6cf4cf699b09345500c90bb90b781351723558089cf135a5a7b0`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-13242c9a12ffb4d788c2f14891b978d17c5b819a44b8fb4dd405e1c1b50322e8 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5365391b6df7fc2fe1fd61101dbae8fa87df185f18b8329f75080a22f10fc704 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-db6a07cea2fa3c1a56c2a19f81a60c909dd757807ab4be94fe07c3ec94c9d5d6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-943db146de733bde8f8d1f4598e2570eddb713c10037fc7a5d96d0ce8931187b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27261c46 | sha256-e5cadca870380aea3f447d3945f2575f032f49732e3ae3534bd0f8c0ee299843 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27261c46 | sha256-00c9f936fa96cb55aa52534e252e9e088f82beedb635d559d5e4ce19ee060575 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-dc5b0d79 | sha256-78e167e4ca72492ec29000949c07d05a58ce64714b4f8147d4a7cf0804f552a3 |
