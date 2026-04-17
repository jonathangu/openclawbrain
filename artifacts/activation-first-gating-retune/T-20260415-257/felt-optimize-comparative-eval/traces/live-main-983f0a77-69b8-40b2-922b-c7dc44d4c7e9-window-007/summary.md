# Recorded Session Replay Proof Bundle

- trace id: `live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13518408454d88b3ad692b956343d851ffe682724dcc9ea68679835cb38cd6f1`
- fixture hash: `sha256-d8ddfc141ca061b024a7735fc1bd6c41a09ad3c89f85b7541ee5a4463459f049`
- score hash: `sha256-6f9ff1dc6be6b48680ee3c573ed8dbd636e852adb48112be192bbcc6d348f346`
- bundle hash: `sha256-b15ab7a79207b3a772a47b113ab0a90830b2f8986b29b66fd94304a923a6bc25`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/12
- phrase hit rate: 0.083333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-76a9870f23308038c7dfa2834df546254ae4769b20da16b32ac7e7ef5f9b078e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-629233d6b0184ed8c9521ff89293e65163c3744d96e8a783391f2fd2db804026 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4c69e4c241ff096708395af316771326e734e343b5ca490007bb2233ed5e3428 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9f00fc08fbeeb8ebb0da84a90541661e5eda48fb633ac89069e662909ca26b8a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8476a553 | sha256-9b8ebeb1e50fb094e9776246eb09a58a6eda0e1cc3b4f842afd448fa617fc854 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-8476a553 | sha256-29a7f2eeb5d68146669c1d6ed595cb2b6b2358440af7107ed0799ce0826a8649 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b606be82 | sha256-628b41627c862e72646ade324e6ef8bca0fa4495344c48db15d37f417c141245 |
