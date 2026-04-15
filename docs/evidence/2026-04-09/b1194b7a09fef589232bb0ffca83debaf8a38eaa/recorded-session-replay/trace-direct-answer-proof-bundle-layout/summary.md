# Recorded Session Replay Proof Bundle

- trace id: `trace-direct-answer-proof-bundle-layout`
- winner mode: `graph_prior_only`
- trace hash: `sha256-54052d000bc2de2ee0ee13e66c958eb9a0908be2b39a2b95467eadc464a02dd8`
- fixture hash: `sha256-ee85abc4e14b2b91215595b14426e9eacc4b89e75df8b01c0a46744b81127aae`
- score hash: `sha256-8dacf4ff6aec6d444933ba68386eac14027fb737e309991af8c285645a65ff24`
- bundle hash: `sha256-e84b53d5f935d350542943dc4f1a91973e121743e43214313a823bc5e1b1f677`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 88 |
| 2 | learned_route | 88 |
| 3 | vector_only | 88 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 12/20
- phrase hit rate: 0.6

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 0.8 | 0 | 1 |
| graph_prior_only | 2 | 1 | 0.8 | 0 | 1 |
| learned_route | 2 | 1 | 0.8 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/5 | 0 | 0 | 2 | 1 | 0 | sha256-07e23608e09214276f8ae4f5271602de4245cc9330c6f83aafcab4e29dab8b2c |
| vector_only | 2 | 2 | 4/5 | 0 | 0 | 2 | 1 | 0 | sha256-b4c99343b23a8e97b3c027345a0042624177b2131d688bc3fdded51ea90fc792 |
| graph_prior_only | 2 | 2 | 4/5 | 0 | 0 | 2 | 1 | 0 | sha256-07bf933d4378ba38ecea714461d19b7959b36cea8d834937bd4ea5d28efd0e29 |
| learned_route | 2 | 2 | 4/5 | 1 | 1 | 2 | 1 | 0 | sha256-4217a5ccfbb4badc668b62efb96cfa3d978bc0f798f07114ae2a0fed225365ed |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | bundle-layout-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | bundle-layout-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | bundle-layout-turn-1 | 100 | yes | 2/2 | no | no | pack-3d2a187f | sha256-dfeba50a7ed731e9a3681e91c489111c427c7be2b8c40e1c8e9bcaa8ea2aa0f6 |
| vector_only | bundle-layout-turn-2 | 80 | yes | 2/3 | no | no | pack-3d2a187f | sha256-dfeba50a7ed731e9a3681e91c489111c427c7be2b8c40e1c8e9bcaa8ea2aa0f6 |
| graph_prior_only | bundle-layout-turn-1 | 100 | yes | 2/2 | no | no | pack-3d2a187f | sha256-dfeba50a7ed731e9a3681e91c489111c427c7be2b8c40e1c8e9bcaa8ea2aa0f6 |
| graph_prior_only | bundle-layout-turn-2 | 80 | yes | 2/3 | no | no | pack-3d2a187f | sha256-dfeba50a7ed731e9a3681e91c489111c427c7be2b8c40e1c8e9bcaa8ea2aa0f6 |
| learned_route | bundle-layout-turn-1 | 100 | yes | 2/2 | no | yes | pack-3d2a187f | sha256-dfeba50a7ed731e9a3681e91c489111c427c7be2b8c40e1c8e9bcaa8ea2aa0f6 |
| learned_route | bundle-layout-turn-2 | 80 | yes | 2/3 | yes | no | pack-547312c1 | sha256-94ad6d75847c3e179941cb2111d53e59656f62efa68be99599493da6eb1af933 |
