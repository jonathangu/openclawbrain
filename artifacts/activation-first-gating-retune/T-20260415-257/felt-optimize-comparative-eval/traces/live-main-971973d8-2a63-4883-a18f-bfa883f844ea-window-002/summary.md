# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2684bc9ce52da3283e7e269a65aeadfa9bb4bda12e0a5937bb82b4e7e3f59ace`
- fixture hash: `sha256-4296b198ad4b2382e867baff61985bd607aae4ddc54e4c60ef5ccb597fc35e68`
- score hash: `sha256-ab380e3d5ca494e56962dc68c22dea85f6f5015c68022a82708123c9f7e2d431`
- bundle hash: `sha256-a808f9714dcac966ea7a752f69440336a34800a3ee1701ea05cf46c4a3b9ec8b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d2479b1210d374fe06946cec83ff362b307da973ce6e0c46c380449deb18879 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-408f7f49e4d5e94c6f6a4cac84ff27adcb9135791fa1513fb570e90b5d953efa |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4ca7610cf79403aeb09ab3cb3270b22c1249169add43f9b5f8bd8d7e556a0a06 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ff13d024aa668a8f31e35fa6ec9e6f43cf01d45e094bc8f0b3611ec031c2ed07 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1af45d7e | sha256-cd6a019562e62b9612db092a47ee7d8ff2c60de7a7bff676d1edff3ac1ee6390 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1af45d7e | sha256-6a4495e3dec85d27d6a9e69b782295bb6f36b1f30d1ac110e58785e50e48e229 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-acc82443 | sha256-3d963a87b4990df06ceb6593c9650cbfffea2006306276849690ae358fa300f7 |
