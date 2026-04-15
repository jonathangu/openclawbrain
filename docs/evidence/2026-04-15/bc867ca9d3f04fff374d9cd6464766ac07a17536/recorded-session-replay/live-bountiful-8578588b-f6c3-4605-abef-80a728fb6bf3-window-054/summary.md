# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054`
- winner mode: `graph_prior_only`
- trace hash: `sha256-009636da02f7f67100c4558c66177e3052d69c0fced9d7e92f816d385fb5757e`
- fixture hash: `sha256-e98441576455ea28abee01372cb0b00d04c2271a6e52b08c0f8f71e05b4805fa`
- score hash: `sha256-2036b1a1356e16efacd8c3815af0247d217805d196ab31db7c02c2f445591bb9`
- bundle hash: `sha256-730bc8e8ab06457a87f5dc6a067f1bfdbf97a08c9bbf5914e1cb881dd50e60ff`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cec99f5e1ad091e90131ca937eb9886122311480b894b2b01cfc694105b3de60 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ab6e9df8aeb56d14885e80e0eff7207755a2144b325782d58c7d347b30755aaf |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8da7f8484080e627bfba54336f08a3b0236b13bfaf5414f1bfc7e55d82d03f1c |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-e726ee088a52d020e8acda97573d19e774c301e72a8057a5989d7ab0715e0608 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-46dce247 | sha256-3bec672f9b21c01847f6f5584c313977f17f6beea404ae312d4ea9993a582b3f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-46dce247 | sha256-18e1d9b3c06bb7d411bf3f9d1e8dfe2720863238b19e7d9e6277b4340de7ca09 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-46dce247 | sha256-3bec672f9b21c01847f6f5584c313977f17f6beea404ae312d4ea9993a582b3f |
