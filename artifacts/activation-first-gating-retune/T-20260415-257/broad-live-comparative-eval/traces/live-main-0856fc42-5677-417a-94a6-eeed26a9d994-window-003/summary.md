# Recorded Session Replay Proof Bundle

- trace id: `live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8112927457240059417bedc3d26ba052a003896d620c2316ad6b12373ef80eef`
- fixture hash: `sha256-14ad40161fa5c35ed07d9d394829c949bb081beaa26c47469b137af3b630df8b`
- score hash: `sha256-8b079c20a76c890fddb2a4e9dab8066ebe81f8bbfc9b59aeeda4805ae4edd143`
- bundle hash: `sha256-e30283cafd8a5a98aa41b3f856f5aef51ba0e931dd58539cdabb14d910663421`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e54eea5dd476d45e5e7ab52a9b0ed2c646fc990677d2858d9966f3baecd8936 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-1768011de9f09cfb20e5279dc890d1122f47e858ef6b7cc03f7b24c28267fe80 |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-360e28ae47508ed9f50451a2136f319ef5321de0dbd9da3cbd6564fcf5fc5e2d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3a2d3f1d6efa741fe9e2e1669e658184ff497ac07b4acca9923e03b44d906c58 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-05b389dc | sha256-37bb0703de732b45ff8fa1ccbd9af844b48beb2f40f181efc5cacf139322a440 |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-05b389dc | sha256-d47b60c4f5ff4a403e6c9577ce6a99ab5724ee2141e8d3c500914d66a84e1b45 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5a38d2a3 | sha256-8c70fef1a6327d328e1d7f363f53d7a483a9a6d097a0449595315b8361f62f6b |
