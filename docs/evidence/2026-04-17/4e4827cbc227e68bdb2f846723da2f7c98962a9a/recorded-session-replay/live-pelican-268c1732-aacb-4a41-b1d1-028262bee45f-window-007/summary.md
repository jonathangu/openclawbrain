# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f65f7ec4c1006917225f6f3df2297434078972719c9016d9a4a28c343601c090`
- fixture hash: `sha256-0846d04b26eef0a1a7c06190a5a1fd4f54e0a1ec3fcf3231ae0df203565132b6`
- score hash: `sha256-8ef4ea9ff1e5e08d2d096382a8d3a5859de0ec536f5686cbcf476d7e9cd03834`
- bundle hash: `sha256-03568a8fd23d550350a04a882e6932f50f646c6c9f7e519027f4c04a32ec3e46`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-42f48b48c6c450f0664e256db3a267d908035a318a1c9a74a979a0b9949d1634 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d706df359343b60dc44df35962aa0332fb9c909c50b74a397dfb62ade3314e97 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-481e76c951f581ef0df807fbd8f17a8b6cfef8d3d497aebc385d35336deb185c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-44da5f964d6a35a13d746d47ba6abf8f75fe32396d28a2fdc94d5ea853425c03 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-90e58ac0 | sha256-738901e9e327446531ccb800f9f057101425a60d2da4a29cdf24d25f8e35a8ed |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-90e58ac0 | sha256-8e99acc90a338c6e047129d29be4ddaed4ab140465a9bcebfde9dada47e07054 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-493a4525 | sha256-f385da0fcd189433f381e72728da22ccf725f8fd2c6143c26386e9306201f52f |
