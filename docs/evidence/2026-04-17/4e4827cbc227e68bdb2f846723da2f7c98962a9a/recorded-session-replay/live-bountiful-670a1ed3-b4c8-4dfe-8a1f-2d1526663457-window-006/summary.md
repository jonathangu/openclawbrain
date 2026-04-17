# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-500bb42a51fe35739e28b1f6be3d9fe7ff92c6a8eeb2f053f3018ae2eba88584`
- fixture hash: `sha256-f69dca5c27c722f582ac3debb2e25adae4c35c5bd6a4749aa476e37eee07c7bc`
- score hash: `sha256-f5de585541c66ad0deb2ce467339a7f4d43e8932680124139b6bdde55bcf6a07`
- bundle hash: `sha256-629d39667cc29cc07d761c6d4d1151e6e5f471f2660c25def97d6fb80fb3b3c0`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a93c94aa4cb26ac67e3ba4bdee5fc22bb0276c3da7ff11089c43e42405c272c |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-011bdc971d001f47b0f2e535a9f1428af85421ec6750e1f3b559652cc4eb23c5 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b0a124880eef07ae0626a028c35d37065818c46656d470166755edb46043269b |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-86eaab5dd735dda82fedd403a69dfe3299cf3cf29f39a2956abbaa51a882ea82 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-67f79e4f | sha256-0aee291fa92eb9d2285376759e1207985549c72e822eeb1bef351084cf11910b |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-67f79e4f | sha256-972e13b0f7e0e827c74104b2df2adf5b709195b9b0fd198da50cf72f7c199318 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-aa197e48 | sha256-48df88b42beebd1bde965fbf99d98ede56b7c9f56d9c8f2d550f0edb26f80023 |
