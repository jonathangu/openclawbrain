# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68b9ce7f8532cca72a3aa69a14e2f86116ae9296f4d403b4d6550bfdd087e76e`
- fixture hash: `sha256-9bea38d812adcb691fd84d63e7a18202d43fb587cc03ce111aeaee7b624c5f99`
- score hash: `sha256-d8da044eb3f886b232449864978058f3052c94cfccadc63fd171bbd094fc7b82`
- bundle hash: `sha256-30aae78bc47888673895bd83368c0f19c8ac703ccbc9f441fa79fcd40c89227f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c60ca09a31056398ac0417340daa846ce032069fff53d8c27abc0aa34201c8e0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-acb24e990f38597bea7f822c78ca184c5f72d428959e8f37fad45b2aad05ddc9 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f369d3417b6d44a594a01aef75b87fd5cdfee269bb73137583768171682b2b66 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-c88ca4cd5238661a012f8f8baf9e6f73b811265809637fd3c35cf3546c091008 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e4696bc6 | sha256-a15a43799a93cde4fc3a7e62b3082397c24f835670a0c806d5377850d9ec3432 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e4696bc6 | sha256-928ec3dd8842cd17d8af41e94ae86d7a5c77fc3fd169d0420de5c057e88bedea |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8119cd17 | sha256-eddb8e8a8ae8ff6f358379b4bd042b30f7acd1b3a21a5572804a08d2e2539356 |
