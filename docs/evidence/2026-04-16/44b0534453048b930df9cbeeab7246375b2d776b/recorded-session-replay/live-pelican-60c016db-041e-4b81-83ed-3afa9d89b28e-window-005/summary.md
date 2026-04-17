# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f4905564adc9cb953b8b5504309a4080c3ac583fe0f629cb62b1e05f91ea23a3`
- fixture hash: `sha256-0ad0b5e1e0f2271069ee0d118e38a8f083b22de4d11f9b10cb9ee63b3ed54883`
- score hash: `sha256-9acc3b7c2bda64737b0f7821ef0fc469b9e124869f7e72bfa33b3c1dcb7fd4ef`
- bundle hash: `sha256-eb6cfe4e974c0fd7433ea49dfbf8b223b251ac880e42c23752298ab31c1bf0cb`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-64badee520388e2e251dcf80ba87d74776085beb63219f4be30791f06cfae40c |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e4672e32b8de322fd72341e7d394d1009ba3115a28aab1abe6f17c8cc2660162 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1068c4548e3e1136bd471f67d57148d3b9dc1983ee847e359f3a516740d7971f |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-ceac8e264b947ad07fcb30f2cd0c619614d94215db8417041c3f9c873b0577f5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-65c4119e | sha256-81c7ee3473213c9554e43cb0ef275755bd5fdb96009154214b53ea0e2076c280 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-65c4119e | sha256-cd49632bf33aed7eaaff711678f93ed62293b666794130b400d6dad98a73f13e |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-2d406b2b | sha256-1ce7938fbb24ab903f654ae3139dbdceac5355e70bafdac45a3afebc496b6429 |
