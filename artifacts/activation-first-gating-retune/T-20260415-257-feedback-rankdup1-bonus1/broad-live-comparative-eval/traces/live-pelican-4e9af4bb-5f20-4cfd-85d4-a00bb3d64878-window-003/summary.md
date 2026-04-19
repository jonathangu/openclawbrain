# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3f86fd026217c7d6458e87e96268ca58f7633ecf498ef1f8793a6a7617c13f8`
- fixture hash: `sha256-c25bf3a6bec00b35ab13366d1787d21cc5e0fb28011aa90689176fbd43238498`
- score hash: `sha256-3d75ca7032724bb77249595bbe86ae44da46e57ab42ac1bdef929a877d56fcbd`
- bundle hash: `sha256-fd440c3ed9112d15b4616359ee918c47403627f3fa554b8f8f5eff0f79214d83`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1e36e95d3b902dbb1cba84b7196a751790c689dc2e631e7340724bc6d85c3a59 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-b3deafd8cc151ff6a92535acd6dd7ebab44af5d9612bca39233cd38684397e38 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-7d119975d76b78c3956c3f6a71dd445e3f7fa1b4f9d004f468b62bb6229005b3 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-c7caec6405a8179549ade532778763c3d62986721558de111afb1e73fa83dca5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-4955dac0 | sha256-517b08811a516164da96e6415d502edcc629bd5917ddc432ea63234cc473a0ac |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-4955dac0 | sha256-517b08811a516164da96e6415d502edcc629bd5917ddc432ea63234cc473a0ac |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-4955dac0 | sha256-7c1c904557ec7979f54ae0c974910cbc86c5d05ddf61eab314ba61e5fc9ccb21 |
