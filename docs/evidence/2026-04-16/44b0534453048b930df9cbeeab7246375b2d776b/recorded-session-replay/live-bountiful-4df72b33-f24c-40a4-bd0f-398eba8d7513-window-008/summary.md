# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70911ec1e7805ccec970087d6c2246db12da18117e08ef4135c17a78ab963e90`
- fixture hash: `sha256-5f3ce437bc5a34220be72a905054c7058ccdfb9aee9afb407a944b39db8e43dd`
- score hash: `sha256-f13d68fa5e102e6c62c6df14f6dd15fc6a3984a5f92214bd276ac243e772ea91`
- bundle hash: `sha256-a45fb63c60367034d01a618fe510f1736a588db44660d079c979c07a9698a588`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c5564d3b460f1097de136a88547e9b2bb9e15503e1a0ceed301551bb8e7b5353 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-38af2db91d9babd274e120f176ae5c2c3202b3e8e149b68e68e5568dee93a637 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a658ecc303b7954e72d174108bb80c33b0602b13a34a7c880d33004ae9f2b24a |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-1a396743e96ec52d95de7cacfb1011e7a4fa3a4cf06bfc661901f194fe26abba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-1483933b | sha256-7278bacbb98cf4a85b367374f477e4688c60cfbc8a910a9faa8b15d2ccde69a5 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-1483933b | sha256-edabe11332c5cef911f632657352099451717e8cc9fd39dc83c2baa6bb79dbd1 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-444718a4 | sha256-d71440f3e2affbbfb168d3d0db812eb51f41680cb8ad5270aa792e9ae78f91f2 |
