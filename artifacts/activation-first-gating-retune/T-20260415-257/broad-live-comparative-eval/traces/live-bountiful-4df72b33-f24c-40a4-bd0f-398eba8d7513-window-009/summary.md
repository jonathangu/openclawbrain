# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-909da7829d18c7b4060630b0679e4b4f9d3623ad878ab660055447ba4071489f`
- fixture hash: `sha256-d6f02df1f7fd44472c7a5dc57cc2a6eccb52d15bdd1b99973bade05111901191`
- score hash: `sha256-78172f57f32b163ccf8e91d702ccb7451ca0058e498a881d4add042b0a586c95`
- bundle hash: `sha256-adbf32559d669ca7ac288464c9ece48cdbac265559ae80e4186e55c6d4dcbb60`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-67d503fa6cc9652ac016339af3a8c1038900f3a66d390047a5f53eeceb6dc18e |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-a99c13e79d5bd542bb1d20736ba86ad74d16c3f0c21b6d9f3f978942840ec562 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c3a1f561e47ce76ba16356d95cac65f6cbf9645b3930867abe28fd5198048e2c |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-a9e29002a63fd1ab858c1a928dba6d1a4198b2ac6264aa9b0b496e2254038e62 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-27482db0 | sha256-241ab509eca74a71246fa9576e0c8ec47ed6cfeccef340d3fcacc2615fd74e73 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-27482db0 | sha256-a4ae610fa1639aa3885336cf6935961c37a4b91dd8f456fec77f215fa4aadc14 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-c03cd5c5 | sha256-3fe36b8df8d4144992b20d0b0922cedb263c37397c0fbfec26d8e9c2b2e97f5a |
