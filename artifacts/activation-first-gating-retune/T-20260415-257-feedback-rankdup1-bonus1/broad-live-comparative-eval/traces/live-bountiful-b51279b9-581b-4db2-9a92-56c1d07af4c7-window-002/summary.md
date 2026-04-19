# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bb16877a0fe2caf32819847f224e56592583ef2fd1a04c845d04e2ee17b64d0a`
- fixture hash: `sha256-4ba8498860fc7c42d2e5ff1842f641b3036471f0db760ed3478d90908a631234`
- score hash: `sha256-379a806b133beb1f33e20e1e788ff7c1c5672c8f53d1e861a7d8897d0fba893c`
- bundle hash: `sha256-6c3a82c7fb8eed7d5810ce80a2eb22a7f30f713ec593f8ed508c22da7009813c`

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
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-09b23957d62d4dd2aef54d6f6e2af1d61d598bff57f4a467b714b73990a75fcf |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-08e58d18386dc4935bed358621c4d8a1e57abb3f327fd59d6dddcdb8db5d6a57 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-f81aa3d32aaca558d05c38a3cc8dd8bfac5a6b9eafc5c7b5b9a04f5045fbb9a3 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5d5db23108c72234533c1ca8d5c7b2e022102c0e8d742323e9e6eff07cb34324 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-873fbc10 | sha256-e4f06711c6c6db0951c62843bcc750a5c9498edfe97798e3ea3e72f863fc6a5a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-873fbc10 | sha256-1454c299e14ee47e1d8fa35372757197fb9c080d504021432dc59f4190049619 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-873fbc10 | sha256-e4f06711c6c6db0951c62843bcc750a5c9498edfe97798e3ea3e72f863fc6a5a |
