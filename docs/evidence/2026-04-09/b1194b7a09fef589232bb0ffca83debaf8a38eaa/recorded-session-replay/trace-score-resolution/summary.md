# Recorded Session Replay Proof Bundle

- trace id: `trace-score-resolution`
- winner mode: `learned_route`
- trace hash: `sha256-574a64bf53c3d6173d64b044723abb88b3517de393d177eab13af71fafd23432`
- fixture hash: `sha256-63d53f199a24fc52c99e70ab08c081d9f795c9234d4c9ad6b641f3f9480003ab`
- score hash: `sha256-10672f3749744d69b3b8eed58c2735df1846633c4b18bca6df1578e3440daf55`
- bundle hash: `sha256-609c835237087343146a54cb3ee081ee638695161a8932cc744977c5d53d29bb`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | graph_prior_only | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 8/16
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 3 | 1 | 0.5 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 2

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 2 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/4 | 0 | 0 | 3 | 2 | 0 | sha256-271076f35fa438fafb2771d3e4fdf49b2bf41b0468ccbbb99a0d1f5bee4f354a |
| vector_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-e5192b59408f5fe76768e47ea616cf29d25978207d203c535d6ee9841b3ff7d9 |
| graph_prior_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-f36dc81d26ed5362af230e6c9c2269f9f5fd6c1cb51d86217ec929e22a30878f |
| learned_route | 3 | 3 | 4/4 | 2 | 2 | 3 | 2 | 0 | sha256-658fda7bb2b0f2ba01e84d25ccaef74c16ec03e2917c4851df896401a9a54a56 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | plan-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | plan-turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | plan-turn-3 | 0 | no | 0/2 | no | no | none | none |
| vector_only | plan-turn-1 | 100 | yes | 1/1 | no | no | pack-3a5807c2 | sha256-0ca721bd5cc18828c36cd2c6f95557869b3d6cf28507ed6b0cf41cba32dcd4e5 |
| vector_only | plan-turn-2 | 100 | yes | 1/1 | no | no | pack-3a5807c2 | sha256-48a9c83c87d436a37475085db8b0a7aa9308060c30e46362ef1865a52d58bf55 |
| vector_only | plan-turn-3 | 40 | yes | 0/2 | no | no | pack-3a5807c2 | sha256-0ca721bd5cc18828c36cd2c6f95557869b3d6cf28507ed6b0cf41cba32dcd4e5 |
| graph_prior_only | plan-turn-1 | 100 | yes | 1/1 | no | no | pack-3a5807c2 | sha256-0ca721bd5cc18828c36cd2c6f95557869b3d6cf28507ed6b0cf41cba32dcd4e5 |
| graph_prior_only | plan-turn-2 | 100 | yes | 1/1 | no | no | pack-3a5807c2 | sha256-48a9c83c87d436a37475085db8b0a7aa9308060c30e46362ef1865a52d58bf55 |
| graph_prior_only | plan-turn-3 | 40 | yes | 0/2 | no | no | pack-3a5807c2 | sha256-0ca721bd5cc18828c36cd2c6f95557869b3d6cf28507ed6b0cf41cba32dcd4e5 |
| learned_route | plan-turn-1 | 100 | yes | 1/1 | no | yes | pack-3a5807c2 | sha256-0ca721bd5cc18828c36cd2c6f95557869b3d6cf28507ed6b0cf41cba32dcd4e5 |
| learned_route | plan-turn-2 | 100 | yes | 1/1 | yes | yes | pack-0557437d | sha256-99c5f98b42ced79e5008d7f6ec8ab6dfae7bf35b8945817db55e200124a57ba2 |
| learned_route | plan-turn-3 | 100 | yes | 2/2 | yes | no | pack-8ab1b269 | sha256-b57d155b7814d4a672b71ac7c5ab6cac55ec68006753577f93bb3156cdae79a8 |
