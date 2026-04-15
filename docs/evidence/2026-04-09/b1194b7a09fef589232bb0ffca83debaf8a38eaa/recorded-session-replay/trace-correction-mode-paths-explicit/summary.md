# Recorded Session Replay Proof Bundle

- trace id: `trace-correction-mode-paths-explicit`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fab75339653bcbb0cb0d29b79ed0413420391a2bac49b3daf77ba52271dbcc40`
- fixture hash: `sha256-27dd9ff4afc45f85fa83146b5f2f7b2ecf21060b7787f28311720e3beef95163`
- score hash: `sha256-861948adf0c41bc28a4502d59d8e74bb0f688248eb88ae2341378e72590377c7`
- bundle hash: `sha256-22543eba7dc0bac15bf5aede6b9d4406be2288cbd89349be691d73d015442b31`

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
| no_brain | 2 | 0 | 0/3 | 0 | 0 | 2 | 1 | 0 | sha256-3bd1527e8f6d8b45bb21926050c3d0ca5b84e1544b44b11e823a7df9dc0523ab |
| vector_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-d745c178f62e61d1e140e4b2f32e794a06308c6c2d50938248b4307bc503ef78 |
| graph_prior_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-f6a41fec8825bad38273f2fc3096844c5cf028282f43967b60d6b316633488bb |
| learned_route | 2 | 2 | 3/3 | 1 | 1 | 2 | 1 | 0 | sha256-5eca755eb9f043620151d3244ab037b9c77eaa0c240251c7c958017c6d44e21a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | mode-paths-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | mode-paths-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | mode-paths-turn-1 | 100 | yes | 1/1 | no | no | pack-d9b1e194 | sha256-e2d63a8d58d87584fab7736d56a1560a3a659cdcf285aa7d1efeba7366911c28 |
| vector_only | mode-paths-turn-2 | 100 | yes | 2/2 | no | no | pack-d9b1e194 | sha256-e2d63a8d58d87584fab7736d56a1560a3a659cdcf285aa7d1efeba7366911c28 |
| graph_prior_only | mode-paths-turn-1 | 100 | yes | 1/1 | no | no | pack-d9b1e194 | sha256-e2d63a8d58d87584fab7736d56a1560a3a659cdcf285aa7d1efeba7366911c28 |
| graph_prior_only | mode-paths-turn-2 | 100 | yes | 2/2 | no | no | pack-d9b1e194 | sha256-e2d63a8d58d87584fab7736d56a1560a3a659cdcf285aa7d1efeba7366911c28 |
| learned_route | mode-paths-turn-1 | 100 | yes | 1/1 | no | yes | pack-d9b1e194 | sha256-e2d63a8d58d87584fab7736d56a1560a3a659cdcf285aa7d1efeba7366911c28 |
| learned_route | mode-paths-turn-2 | 100 | yes | 2/2 | yes | no | pack-32f4a2d5 | sha256-298c8349fea80f5c1a86bfe4250f080af9a20a6d7ef4200bb887bc15a99dbc5b |
