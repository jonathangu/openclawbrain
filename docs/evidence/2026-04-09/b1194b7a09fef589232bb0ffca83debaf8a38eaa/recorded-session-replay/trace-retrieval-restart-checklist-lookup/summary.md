# Recorded Session Replay Proof Bundle

- trace id: `trace-retrieval-restart-checklist-lookup`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6f5f930f2df03ad616e08dfd2f0d1e71d1aa99ff5985e8434e9947ee2ee26b92`
- fixture hash: `sha256-ee04d0d989fc57e44664473fe8d656d4383122c7b35b3a2ebbcac2b5c1400aa0`
- score hash: `sha256-90c8f852831153b110e2357447665c6e2192116c113f0e0088a23f280221bfc7`
- bundle hash: `sha256-d5201af94c012c2df5ab8e6f8c9096011bd3fa3736819d9e9d257b619096360e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

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
| no_brain | 2 | 0 | 0/3 | 0 | 0 | 2 | 1 | 0 | sha256-d4a1b84ad22f95796c1566af294f5d58d89f7c13c9b22da34e8c8cc232f5be7b |
| vector_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-7889b33b5226f2653c7e30e85efb376c97bc201450fdfa6a14e13fdc18a649bd |
| graph_prior_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-da8db0183c8875c4c3f485ad1e8f7a7b561fce9b3276db95f01e7ee85f5749b5 |
| learned_route | 2 | 2 | 3/3 | 1 | 1 | 2 | 1 | 0 | sha256-fe236040be6c905d43b92ef4ec9af6cfebb8f5b5cc86906a61b1b55a030cdb32 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | restart-checklist-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | restart-checklist-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | restart-checklist-turn-1 | 100 | yes | 1/1 | no | no | pack-521559de | sha256-74df6f718b3b9a7bb47fda3c79b67527a8639cb66d32ce98a15ca994e56fed5d |
| vector_only | restart-checklist-turn-2 | 100 | yes | 2/2 | no | no | pack-521559de | sha256-d578ae8c9cb0b5579d0a5321c0b1d0ca317fccd38612f19ab75348bee8eaac0e |
| graph_prior_only | restart-checklist-turn-1 | 100 | yes | 1/1 | no | no | pack-521559de | sha256-74df6f718b3b9a7bb47fda3c79b67527a8639cb66d32ce98a15ca994e56fed5d |
| graph_prior_only | restart-checklist-turn-2 | 100 | yes | 2/2 | no | no | pack-521559de | sha256-d578ae8c9cb0b5579d0a5321c0b1d0ca317fccd38612f19ab75348bee8eaac0e |
| learned_route | restart-checklist-turn-1 | 100 | yes | 1/1 | no | yes | pack-521559de | sha256-74df6f718b3b9a7bb47fda3c79b67527a8639cb66d32ce98a15ca994e56fed5d |
| learned_route | restart-checklist-turn-2 | 100 | yes | 2/2 | yes | no | pack-54f4b15a | sha256-6d44ed6ff670342384e786e4496a9c88640c79600e91c93425df880b6658ff8a |
