# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fae163deba3753cd0d67293e5b3321d395cfd004b4c89b21f894e2a2672460a7`
- fixture hash: `sha256-3c60ee2a81318b7745043c018d8fbe0ff4db3777c3e35e5851f7f1b82123cf0f`
- score hash: `sha256-84d74b641101f2ce1dbfe6478b6d3b97d786c2f7110116ba53d8d0338b506ccb`
- bundle hash: `sha256-5ff48f36efc357d3a4531b80b25b96fabafa3a0f623e48fffe7f410030fb5750`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-50883b87c06b74bea3cfb62ff5f8bc47778790de2af8fe623d57526aaf2ffdc1 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-2cfb520eeba32c8c2f7d0159fc9c3d1f374de15da89ecd3ed7fe3a0d79d93b77 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-615ab415d756e07c7f58a16b15d943009e24a0e75c46820f1d5f5139ccd350cd |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-a06fea5c8fbd7b257ec8fa4234ccfd3c7a2187996bc556d86cd25925482b8edd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-1ad01202 | sha256-16544a6ff049cf7a16cc18fe35ae17a4700c47f1e59da0afb90bdb1ea6eff1bb |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-1ad01202 | sha256-56ea6feaebafae611f584692014c4de086bddcc2d89c91a7f1a80d2436753053 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-19e74c2d | sha256-2ce5a87b2eac73c5b5ef2aaed52a11b7fba209c1bb1f1b7e5bceb332ae59e698 |
