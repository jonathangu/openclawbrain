# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5176937ed6fc9464e75096c1c5b5dd165c66db577451f32573da064aacbbd206`
- fixture hash: `sha256-c4ac3ff5736ddcdc283a0e74e44fb3cc2c8ef5acca6ffc633b4bdf3e072f174a`
- score hash: `sha256-abc6633409481d99b9e15cf9925735dc72cfa811a72b56c571d00ea6cf530f0e`
- bundle hash: `sha256-d7f7c320c7f2d34027776efabc569096755ab7e6931d4fce1db4a5489bc639f0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-303b09693534000a07eecf53da8c48518fa490f57e0684dd58a4834c42c9644b |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5b4175996781da08f3e2817e626d4c35152552ef041ea886f352820cb225c67b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-38a3a7adb45bb8d19c4f42466fdb742fbbd1d9677af526e268b38da8ddd67ed8 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-4da10fefe6308febe9a71bbf84efdef981a29fc96e01173fcd2803c1363ccf41 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8cf3e154 | sha256-af3883b3a1377a539311619b0f1cd182d394fbdd01f14163f26b4fd070b54c90 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8cf3e154 | sha256-6221492584a3cf68421c3ef25cf124410b66ad74dd7b315839dfadcdb0b52caf |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8cf3e154 | sha256-af3883b3a1377a539311619b0f1cd182d394fbdd01f14163f26b4fd070b54c90 |
