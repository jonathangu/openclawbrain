# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3f86fd026217c7d6458e87e96268ca58f7633ecf498ef1f8793a6a7617c13f8`
- fixture hash: `sha256-c25bf3a6bec00b35ab13366d1787d21cc5e0fb28011aa90689176fbd43238498`
- score hash: `sha256-8ed2829a96e9146d90ff535dc78202b60723159d3d7d3773159b1198b9acc2d4`
- bundle hash: `sha256-b028664793b713020df2e2e55864a7a20b8320b885961c3d1ca3c47fa2599141`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1e36e95d3b902dbb1cba84b7196a751790c689dc2e631e7340724bc6d85c3a59 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d523c71b035d42cff128f5d370f4937c750cf4d8823082e5c157d60000774daf |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-09a805886947b07bf158008b76e95b69d04d88b0c003ee7af10290e8617100fb |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-2b8ae7bab79e343b2ae8cba8b051fdf299d3e2f4344de2f2759cbe7b527b5241 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c3e3d2c9 | sha256-38758c053502ec4b5b52e22f4f4c50d2d2546ef360b5473b12811fedcb757333 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c3e3d2c9 | sha256-05baab3d7b1f15c71ee9843981dc79447e524779f48a39809e6842872962898f |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-c3e3d2c9 | sha256-38758c053502ec4b5b52e22f4f4c50d2d2546ef360b5473b12811fedcb757333 |
