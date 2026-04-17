# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a6c28313900614132b45e0ea565c05af51ca312d324fdd7ea457443cc9732010`
- fixture hash: `sha256-33c60ab5b3d1fed7da251c9623ad91d7f552d5ae6e358c91f60c002d0d9e41c0`
- score hash: `sha256-a55b3a78233e29b867dcfbf95d3cb910e06fbfd44e341ef432d14781dd221d6d`
- bundle hash: `sha256-3b162fba7229f36459481f4b16301b44257cc3e6f9e6f19c97fa40d2da3c0b5a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-43901bec518b9820aa1206ca8d4af0fd884bfda276fb97e746b980a65cc6c82b |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f2d98ae26d9fc9bc56c24b5490a04d43389553467525c87f509d3bbd5cfc9094 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1b2bb5ab7421885ed612e1bf108163d347d3f74d040b61473f008ac895771799 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-9b43f4676d873d3b574fdeb351a7a807a1ca93ae3a8461b92535566c003a3343 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e4bb3ed9 | sha256-12db4261fac0174644f0d3ada700ef8ff80f233efe7b05c996a24b1cb657d93f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e4bb3ed9 | sha256-5e3c2191a24492a724d2ab33189265f28ab4f799993f734cf381954586dda8b2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b60ad0ec | sha256-57998b94bc30b3107e7e7d90c1ed615199f54d94048277f85fc58dc05a17f45e |
