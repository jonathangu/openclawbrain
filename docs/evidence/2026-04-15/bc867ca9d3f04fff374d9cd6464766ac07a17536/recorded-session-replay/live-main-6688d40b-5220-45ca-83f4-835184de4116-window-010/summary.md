# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9af4d3068fe0abcd8b0d002d37c1f3cf1f47d195e7f9302f6c99d8ac1c1ba8d0`
- fixture hash: `sha256-b1358b7f23f888234738e0f7490e996569d9c09e6a59451858fe36290e0374c4`
- score hash: `sha256-757baeecf408528747e8cf5cbf0f5f4570ff80caf51ed642c0026d5567332153`
- bundle hash: `sha256-e2eaac0b0f21dff18b510673c77646a8f355ead7662be47a1a590604c1c5364e`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f9d25aa12eece409644e17318b585516e6b4b6b447bae003a14fd993e376b98d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2c380de0bf5b2bfe0ef81c3149b881581d55e680d7a68d2334b508d6fa1a9473 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-861035e5f5aa79a542cae9baa2bd774026bd951d8c94336311d219496ad6f363 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d026ce22 | sha256-f13110415a0447f93845ee3fa2204308d3fe499433ceeff8f71f9999a174f9ad |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d026ce22 | sha256-651e8104ec10ab35907904c9e4021e82f5de66b33692fc7f22e1c2e857b6c02c |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-d026ce22 | sha256-f13110415a0447f93845ee3fa2204308d3fe499433ceeff8f71f9999a174f9ad |
