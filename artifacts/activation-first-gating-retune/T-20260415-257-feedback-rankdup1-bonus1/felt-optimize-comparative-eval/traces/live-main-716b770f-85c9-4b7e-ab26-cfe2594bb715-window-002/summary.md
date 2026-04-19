# Recorded Session Replay Proof Bundle

- trace id: `live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e321442dc8033dd76db95133894d776ec05ebee5a5a98eec612f6b420b907658`
- fixture hash: `sha256-742118fbdeeb061b08c45664c524844d158f1b6be0af589fa277c4ab60f660e2`
- score hash: `sha256-ee330413b5edd95838a0450eb0e53fa5b839d467d9a39d38043187ec79896a29`
- bundle hash: `sha256-078656a9706e3b2cd5d276f18b9fb5928dd0113d47aeedbdd0dfdc7687c8d574`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-05b0912208f70d1fd8d2baa8f914bf08175b3f38b8f85e68cab4f50d835557ec |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-6c7dc23ea6c0f3da6f9dc9888823fda4ff3fd157191de61a9aa34e41b704d2a1 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-5e05a647e429ecfe15faf433a0d9644b1cd79bb24d101de884acc067a60f0852 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-b77a9c7a3fd050b98645168b1be756ca8e6c5953ae851e3df40cf768202f9e90 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-00b45518 | sha256-4ec1526f94c34022f6fd2fb7b968ed45db9fb3a563efe53d0cdba46f868cae61 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-00b45518 | sha256-522190eb7d6a7e6cdcc47b3d33a151448d48dc73306d17a62b7d522c4ea18b3f |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-00b45518 | sha256-6f4fc1450001333c7337a34510a398a10f15ec998420873b25e47bab6c5825e4 |
