# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b29852ddcae763768818b925bfa513423dbca5c8ad934450c78f0838b90cfab`
- fixture hash: `sha256-f828a5ef63881667b78ea5f5530e5417bb5590176f57bdcf8c4590150136788a`
- score hash: `sha256-d83246b92b9ebf1ff7400c3a4e9d51b6f8fa49fd4650c3aad989bf4dca0f89ef`
- bundle hash: `sha256-325db4d251ac3fb27afe359f9da9e0d41229d8eccb51dd88931eb31cee882cce`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d23197d4b519cff22649347398dfab9ce049fcf294afab672a8b41fd8ebcbbad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-931cfda16d6aed75e7d79288f1f5601fcdd21e6648b13dc211bd4641ceefa21c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-52b8054158617be487f671788c2de2b6d43a774142e388b7de100d1e51f85e75 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f82a86510fcd6ca74221d84371a1ff021e38769c9d39e06f0d5e9b8ff12ace94 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fddfa59d | sha256-a84254936904d7ad2bee7ff4023eafcc8482d72f88ce3c5ba407245b8b86abc2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fddfa59d | sha256-bfaf71e258a306981b4c7939488913343243e28a47d77bf1e0d3df85ce2eb46b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-bde307de | sha256-a6a8beb7bad653b21ddd27a7bbf589737f46efc02b4ec41537aef5829ed8842e |
