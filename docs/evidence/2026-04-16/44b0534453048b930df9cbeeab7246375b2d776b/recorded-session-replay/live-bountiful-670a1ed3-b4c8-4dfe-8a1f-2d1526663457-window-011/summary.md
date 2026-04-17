# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2d785c91c6c2597c88bfdefe91898000c30733ecf3cca8e1fa5fd2d6621049e4`
- fixture hash: `sha256-63b7942b83cea800c5fc9cb957ce0307322538d9d8e1a745ea7ab80b74e65911`
- score hash: `sha256-4a145559c8f8a8364881e5e8acd831cb5c773bba16711df9e2037170fc9b9fef`
- bundle hash: `sha256-a189c15a28798c07bbb4f3fba1e2987f3f630484532516dc94a981cc53d8f7b6`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4648c104e20ab98d8928f41590949536cf65a6240f7fac95811ce6126bd169f5 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-f92c63c5919965a1c5141397a26f10fdaf20f3e5b242e224e2ffdf61315cb0f6 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-6110df31525c9534437381d7e72c5b7443da49bb30a203163ebba3d6a561f4d2 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-cc39b9bf98b7613ea73d1399df20390dceedb1c59e1ba83d14dbe72fbab4fcff |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-da64b230 | sha256-55b7515078c1f26754fbda56967d11a85c5eee9da881bdb440d98221adc27637 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-da64b230 | sha256-9b21b71a4048a6bc2363406fdb937406fc5ba94c5898925d2498c634f7f15f5a |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-3cb70e4b | sha256-d7901e416d19de2bfe8b57c09aeeaf8645de35c37d819e996c93adc3e17b5308 |
