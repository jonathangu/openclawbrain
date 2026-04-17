# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98ce4509785da1d3e9688496a53303f79675442a91eaedda79bdab30b5e6b8cc`
- fixture hash: `sha256-ab905612bd3cc43deb68d413a855b981990f021bcff6e0685761c3af602b59e1`
- score hash: `sha256-29038e0c952285835192d41998d14a02bd37a42b25119102e077fd168f96392b`
- bundle hash: `sha256-3fcaee97128282b51a5ee7b1c03878d44bfa099ee0f486fdf30012d21f67581b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16539ac70abd2ef9678c6c7835bb8d35322c600e9de7b2b4d16217df707851eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fb42d4318f88e825133f2bd37ca8e44858a3ba5ecb8e25e0aa99dffbaaeb9e07 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5d3e20c9904efa91276d35b0b6b3167170d438b40e92a7966af2d0c90fce9f6f |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-0ccbb71af2bc7f97f8f5279e42798bfb2998563c82fabb81c4b5868eee3bc154 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-642b5942 | sha256-ca1f4b47dd08a0a3f1286b8e330c0f866c72ff7eefac122533717aac28123b1f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-642b5942 | sha256-495d8db1ff35817cf6aabd6e908973e0d0289b936c9f5675a668e0b64beb2fa6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d2dcd6ff | sha256-cbacb48ec2d4221d9a8b0d7d80a9544570d6c29a1ffdf8a4f96ab94bbe66ab45 |
