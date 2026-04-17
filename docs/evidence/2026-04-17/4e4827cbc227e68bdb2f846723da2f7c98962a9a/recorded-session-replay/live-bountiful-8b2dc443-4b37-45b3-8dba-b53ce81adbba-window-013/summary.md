# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-07e10f6820bf810e1999011ea58316d9d53ec99aa0ac7473d30a2c9a79d153ae`
- fixture hash: `sha256-54c4d68b5e528e2dc7ad50c599fd75e1b659a972d8d4c97376e292a3ef62dcc8`
- score hash: `sha256-a0a35d02f09bf57d173d6046c9255a76f02bad8e04fed15dfc7e72b75ff51380`
- bundle hash: `sha256-f20e0023ba6c233e0fb5021cb0311972ea0341903666bdb942b91b79a85ceb2b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-430dcaca40205cc8d42bfba95521d8acee2a6e6c074b542cd0b9a2d9f1547939 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7cd7389ac1ecf0b62a1775539d4ff23a785f406590a2eaa7345fb8c72c680ccd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd3d8022166d5ee55a004bbc2bf11800e6531b7aa71dfa0fb89637ae1b6496a8 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-674b0a0ec80f23d28bce9acb08377e2473adab57d3043dec3ec0deacb64f813a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8f203624 | sha256-1c353fb803983c3fd1c248414860f74c3e51c0f4e002f149f2a181165c43e439 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8f203624 | sha256-c6858dcf4462f827f8966b5f0d5737bfa7600b6bad9ff575e5c8a0a6497ec930 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-544b59ff | sha256-af362d5b526877fa124f20d61dc91c1608e6f7007302e336952e5d531308b6f5 |
