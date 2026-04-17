# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9af4d3068fe0abcd8b0d002d37c1f3cf1f47d195e7f9302f6c99d8ac1c1ba8d0`
- fixture hash: `sha256-b1358b7f23f888234738e0f7490e996569d9c09e6a59451858fe36290e0374c4`
- score hash: `sha256-a4449827185aa65b63f35e3967242c55c0d1a3da9058bff7764c20088c9c80d7`
- bundle hash: `sha256-a45a9df73f126e6403fbd55994e8464fdb5a46f75bd2e68afc554bd5b74a64bf`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-38fcd5e204f9ce44394f224032307e552dcb6c83cc2ff3a9c8d07b3df48aab19 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-73118df9f79ea6e64efb80fb1e40c8eb2d993511b16a60b5a533851d806396e4 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-36d0a905d4232d0b4ebd3d0a8d93b5ae5f2b7bed6fdffdc5e22c3a61a454abde |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-6f849ce69eaf548e166d5046a22eaa8904531c783e426e6088508f562ac0a2ea |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b42aaa81 | sha256-1f2ef612aadc73fb3f8c780c2f3cc0303ede64d7b198a6c99895c00a47ebf231 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b42aaa81 | sha256-5407e47c0b4729422dffcc839effe94e1044bae9c1a2b7c9d34886e4ff106e43 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-1da1708a | sha256-4b8771b05a858d33b6b73ce3db6d32ea365cc895ae36a36a3a580e39b8058c09 |
