# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049`
- winner mode: `graph_prior_only`
- trace hash: `sha256-10f30fda1583220ffcb0e13cb73de4976d5f3f5f0f058e8e816ab9eaaeb4bc0c`
- fixture hash: `sha256-81aecda5857d0ab09faf0a56bf49fbe289e64582b0578df3f1535d5bf05ea11e`
- score hash: `sha256-a446cbccea9c5e67cf29b84004679bf600d63cd799f6089ee158472dc47cf4a6`
- bundle hash: `sha256-3c43df8b4fd79b29e5f2ecddd4c2a9b1a10845c45ab239ff3f368d65d8d333c2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd49736decd703dddc6036cdf0bf744059f6270cb8728fb209a65a281dd21058 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4ff4057ddc1027c44af0fc11233a1d41e8a2e4030d3853f3c2f0ef0895b429dd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0afc8faa91baf0ea5f66a2c20862176078255ba090dbbf9182c805121c48864 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-bf4e2b60433e18c8ce7aaf6be61aa913b67c63ec4bb2d303c3c8d8cb0b0f48df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0fcc80d2 | sha256-a76197c50929d41d2489a1389b3e7a3f60f370472124bd701bbd9845206cda71 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0fcc80d2 | sha256-4ffc26f6d0bf70ec1913ebe2e74a6d9fcd4dc4aac93b804d6cd780e0b97dd2a5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-6df6554d | sha256-df6e8ca2b90b1b23e5fe5b6c901abc342d049567e51ad53919e87469d513cfb9 |
