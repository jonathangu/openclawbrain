# Recorded Session Replay Proof Bundle

- trace id: `trace-correction-rollout-verdict`
- winner mode: `graph_prior_only`
- trace hash: `sha256-16b1c9786508756f0b9bc0745893317b2032d3d7f42af8b674ab6c96358a37bb`
- fixture hash: `sha256-33cce9db06a1c0557c61c637734ec6566a42558c018e5cd2eca45eb861553334`
- score hash: `sha256-34e9004bb8e64367d20ed45ec4221f73afbffd7fe6f08eaa6319d559082a1a03`
- bundle hash: `sha256-0c632527c097c335e9c0bb57fc80671fb36032be5d70725aabdc891ea1315f62`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 2 | 1 | 0.5 | 0 | 1 |
| learned_route | 2 | 1 | 0.5 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/2 | 0 | 0 | 2 | 1 | 0 | sha256-d1c39961f2988b0187ccc97c6c8d166f325a88a382093d64e508703691a75ec8 |
| vector_only | 2 | 2 | 1/2 | 0 | 0 | 2 | 1 | 0 | sha256-cb93443e4de7f676d1b2f184b3a5680470b6d56a457d4d3abf0343bb6db5ca77 |
| graph_prior_only | 2 | 2 | 1/2 | 0 | 0 | 2 | 1 | 0 | sha256-1f16f9fe4671f202925b5646976c2e34f79c047da51a02d36473e71d6f0c3734 |
| learned_route | 2 | 2 | 1/2 | 1 | 1 | 2 | 1 | 0 | sha256-3146ca02a0dfe7ab66461795210439ec68cdf0e0657c1103feb562eac677a6af |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | rollout-verdict-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | rollout-verdict-turn-2 | 0 | no | 0/1 | no | no | none | none |
| vector_only | rollout-verdict-turn-1 | 40 | yes | 0/1 | no | no | pack-c4cc9128 | sha256-1180aab7b0dc8a11c518ca48d596be8d6acd864b71fcad2c56eb2a89d4d12d7a |
| vector_only | rollout-verdict-turn-2 | 100 | yes | 1/1 | no | no | pack-c4cc9128 | sha256-1180aab7b0dc8a11c518ca48d596be8d6acd864b71fcad2c56eb2a89d4d12d7a |
| graph_prior_only | rollout-verdict-turn-1 | 40 | yes | 0/1 | no | no | pack-c4cc9128 | sha256-1180aab7b0dc8a11c518ca48d596be8d6acd864b71fcad2c56eb2a89d4d12d7a |
| graph_prior_only | rollout-verdict-turn-2 | 100 | yes | 1/1 | no | no | pack-c4cc9128 | sha256-1180aab7b0dc8a11c518ca48d596be8d6acd864b71fcad2c56eb2a89d4d12d7a |
| learned_route | rollout-verdict-turn-1 | 40 | yes | 0/1 | no | yes | pack-c4cc9128 | sha256-1180aab7b0dc8a11c518ca48d596be8d6acd864b71fcad2c56eb2a89d4d12d7a |
| learned_route | rollout-verdict-turn-2 | 100 | yes | 1/1 | yes | no | pack-26b0580f | sha256-2177912b4495e068e9df65cbbba89dcd187d3d59e759bf60dfbfc271c126cb45 |
