# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f9aa9aeb2a008ffbbef66937498f659450de790103271a3013e9525a14c6fe94`
- fixture hash: `sha256-5a27682864273526a5ef1ec747be28d22cb7ff7f18b59d5b0629943c5f759e11`
- score hash: `sha256-41c7b768e9b9b02a2197c9f4a34f67bc09e977106fd93413e4ecb5b3108b1529`
- bundle hash: `sha256-fa2412b3e38cf5a5e6a322c2a2db2b6d817be02309694f0b7aedbee8b4f0a76f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6b2d51f37ef17f0ed82a2f36897126b205c47228efe0e37855cb029004034490 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e7e96eb6bd36e60cba0e209edaffe80c65f59b0850b89f69b18283ccf1a2d51 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-2ec1acc7a887d3741de5dd149b0d7039a8249b0e1f9b7497fcb7ecb5adfeea90 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-bbeb73de2b866027e10ed9c1ac0677cbdd503a64bfd88d1ef93e719f18982631 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-f342e15b | sha256-cbe28a4e4391659c3fd9a6addd89d873d4c3d67f8b00baf12a423c0dbe1b9227 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-f342e15b | sha256-3d4ec43953557f0781da65a9e80baf6ecab2817c4db8e0ef48636b56a5d8c712 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-58439af0 | sha256-8c16d59988995e880921409f8a8d9f016edbe87e6022439ab32f2601d6e87cc6 |
