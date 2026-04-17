# Recorded Session Replay Proof Bundle

- trace id: `live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4d327d918fbcc4a369abe7ef4d164f9a5cfa04faf7adb2505c432935e6de9ae6`
- fixture hash: `sha256-6153c43221a8c0bd8b8f42dc9046e70b2d1a03e5bc667d5e9fc62b4aa1f0fcb9`
- score hash: `sha256-7d1ff56748aede5054cc3601906fa1e9aaefe82040779498be9f9335eec639d4`
- bundle hash: `sha256-74da4bf11193851d05aff37408034376099345f16a611f4e0a7c84140e97df6f`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d542a8f0c800204dee4f72d85787e8cb1b923c865594f3befa573eb5cd2d9388 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-804362f1f46445c976aa8e805422735784e6a0187f022360ec2c5c9dbc84b8b9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a496bcb7ba6f9c851ba8509cec88e015ad033e9129c4a66dc8d116e4721622ff |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-21e09004992437cb25cadf4de4adcbc0b47f4386bbc84dd0fcc4bf2c1c5ead88 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-29ce77c3 | sha256-2a0325eba2d80fb9cbb7b08d34874ab8cc5c172bf729ce2d95825feb7a9cad39 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-29ce77c3 | sha256-0fc32dc0a3581e82936728d6953848983ae92a3291c492ee18d5731da727f5bf |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-613e6474 | sha256-0ebbe242c3717d51dcc4411c775059b927067ab4c108abe353d1a93745b460b6 |
