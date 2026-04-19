# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80a26a379e19395f27acf926c2152bf691413288f74fb851622ffb175287659b`
- fixture hash: `sha256-b5c7f982a4c7c837b7b862133d2b0fb112246d1e5da8e79088a56147358f56fa`
- score hash: `sha256-dc4ef25689aaba06edcd107fce84926f04d01e6cd555f6d0f36f274c967eacd0`
- bundle hash: `sha256-3cc4629e2debcaeb64e62c3f5d349131ea7c3df98aadd6d3e30b1e77c82dabe9`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5e03ec51b785f9aa9cb42beeeff27828af175efefc4b17a7390a8051e0981fb1 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-683f081430ef80096ea4ae08095b617feff2f80bdcb9188bfe0ba37b0ed8bbe6 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-fb2c8b2a824672f1bea3a9c67cc44441be844703e04d9f94dd6cb33bafb3cb33 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-23ebd51abd2e4ff05053d66b16452288634ea502cf5205d93dc2221f5eeceb02 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-469572b3 | sha256-f780951019ff9e67696ae3f0a0a5741b97989743d0e0b942f8d098438198c2ec |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-469572b3 | sha256-fb4a6def88c6e5514bd5d1643eb020020f59b24bf26f5beab990cc69beb0fd5a |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-469572b3 | sha256-f780951019ff9e67696ae3f0a0a5741b97989743d0e0b942f8d098438198c2ec |
