# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068`
- winner mode: `graph_prior_only`
- trace hash: `sha256-96a8ce03d5ccd3eb8ac2b891590e50e52a93852936116375e423e4aa54e6c87b`
- fixture hash: `sha256-5bbc260fcf82f9c3549879279567f4904231f7dfa6b4db116db4ff63f77dfa74`
- score hash: `sha256-fd09b3de1a9d3b18ac9c2f5e4ca568b9e7974acac7a4b49d8ad92964d7a52dda`
- bundle hash: `sha256-02ae80bb8e08a8c4feb2ecb3f90b5897bd48a5a4571932d281f6fbfba3024fb0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-65d66b9ffd94fa10d6d8c747df6232ecc791c54f9b91e74c97206816ca5781ad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-00b9dcfff33a324e7b77596376578c264776dfedb83a95a6543ab652302972dd |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2359a80a9c95d3bbfcffacdb1fbdf56213e68fae7381905be9915590b6a0eeb5 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-55c2c1fcd89255242b159de3652f80a361952b302cd8ccdc8932630d41d38d77 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cd9e63a5 | sha256-2d71a42f57e33bcdfb447b8f94a58df22f36a3aaa51a088399f25908c287ec95 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cd9e63a5 | sha256-934e537549087f46926fac4cce028ee2eca4912ac42d26847d58384906585462 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-cd9e63a5 | sha256-2d71a42f57e33bcdfb447b8f94a58df22f36a3aaa51a088399f25908c287ec95 |
