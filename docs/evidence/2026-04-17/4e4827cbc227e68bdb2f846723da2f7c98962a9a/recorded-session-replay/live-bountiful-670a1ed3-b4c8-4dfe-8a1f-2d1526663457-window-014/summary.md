# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-609a8ef08a0005ac8c6d28613bff0743081fda2af2229951c1bce5c2a71dd05c`
- fixture hash: `sha256-6a13457eafa6a8dea8911b77d2fb44eb3c714588ecda4ba2d46120f25504eae3`
- score hash: `sha256-74a5332f4ba3b5aa0c4d9480760a0dca69eb52c38c84b6e8b26f4961461bddb5`
- bundle hash: `sha256-d4e4d6e70fdc3b2c96319a53aba38b838f0dea424a39a114789b7b29f8078be9`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-529cbc550c979df64d974591190e0c1d456cbd8f7265be9e27a0fc5cbc417683 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-fdbd91bb08022360711a45038b078a971ba2095cce92a09070ed67a8c61eaa3a |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-0f35a09c31aa4ac3feb3e3d7b8f48807b71ac078feef7a329f92e6fe6a752c5e |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-9fecfcb41ae64e0b63106d3104f4b3f61e35c692c7f8d05acdae0a88f7230328 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-4814fedd | sha256-f6dad16f48544f8f757d3eef34040b5bdf799cdf5c0c7fdd4c05f40698deb8d9 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-4814fedd | sha256-3f77602387d9267ee778dd5d083e8a58faf4f62ffbd4f7dd6bd0339d0e4e15cd |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-a1b591fe | sha256-4c10d57286847580ef10bcf8951db5617511804e067316c8b5520fd5b52ffc74 |
