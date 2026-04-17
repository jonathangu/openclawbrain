# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9ea17f3aebef4af75f0a93c521d6e776070c7076063ed113dca780cff0b9684`
- fixture hash: `sha256-e43f09daa5c7f1f8012274d4f09baa27758aaa51c3e914baa4ee6b5329b895af`
- score hash: `sha256-71267ac7c8ce985a0198a150359beb4bfdaccd990ee07d4a48fa80f9be1535d7`
- bundle hash: `sha256-c024918707d22e8caff522ba0077ba4f05bbd2ea64f3b51668a122190312e430`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-174fe02cb9d576a687ddb560851b02ab0e12cb6737fa301408229ad552fa41d4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9378a7ec673e5c13acfd1a84a41e40c8eaa449bad60041141a60d87e79d297bc |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-83211f7869ab9534124e3232dfb6d7e87659a9e9598a7ffe0f93201c20dd2705 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7f099b3e09ba17bc67621d4e15e3272c50290c2758541665aae3e486eef16987 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c89dc294 | sha256-a8ae5151da4795e47b06e9f41c82ca57e1d039c7e69d02df11b6cf7426177b8e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c89dc294 | sha256-943b94384a4eeb2570d1bc0e26da14a9068a06868ffef2364238a76771bccd36 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-537674b9 | sha256-99c7b7342233558f342db5fd3945875fbdc369ef916879943f0ee51f43c466de |
