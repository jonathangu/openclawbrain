# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054`
- winner mode: `graph_prior_only`
- trace hash: `sha256-009636da02f7f67100c4558c66177e3052d69c0fced9d7e92f816d385fb5757e`
- fixture hash: `sha256-e98441576455ea28abee01372cb0b00d04c2271a6e52b08c0f8f71e05b4805fa`
- score hash: `sha256-88d5399f077544fbdcc5d59c4b2129e48e028e86cd60eb2f8e119b039852c36f`
- bundle hash: `sha256-3942fdd06afb09a4d3f294f4f1f573fd109492e5a7a1c203f20f0ce8ba42c116`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cec99f5e1ad091e90131ca937eb9886122311480b894b2b01cfc694105b3de60 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ab6e9df8aeb56d14885e80e0eff7207755a2144b325782d58c7d347b30755aaf |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8da7f8484080e627bfba54336f08a3b0236b13bfaf5414f1bfc7e55d82d03f1c |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-50134e7d50979f90bbf069d08af184350505d2d756982cbe9c6a44f41b242be5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-46dce247 | sha256-3bec672f9b21c01847f6f5584c313977f17f6beea404ae312d4ea9993a582b3f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-46dce247 | sha256-18e1d9b3c06bb7d411bf3f9d1e8dfe2720863238b19e7d9e6277b4340de7ca09 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-a02ba616 | sha256-7614e4b098578b3d72ad867dc63601f833a4a7010b9ea3a307caf14f6757a2ee |
