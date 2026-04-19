# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-caf66a8ae0f50cd2a6d3d3cfd5e877c0e6e80108a88f78a99bf9d0af8e14c2ff`
- fixture hash: `sha256-d0e5149f0ea8ad48a690b98cf321460b1ba9a083ea4dc63f8728a3baa728b105`
- score hash: `sha256-1934f98d0df0c01e467feb8b667f959088751923777188bfe248e3d0e47673e8`
- bundle hash: `sha256-e0575ea2c85087d90fee3c9339ed603e37729919fb013dff6a1b94a30d75dc4f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 1 | 1 |
| learned_route | 1 | 1 | 0.5 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9082b7d02459eb2c821fe90896dfa510257befaff2be31c5d25dfe86c62c130d |
| vector_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-46da1ba9aa72eb9ca7274ac58377d44c697799ab0df040f9a21da69588033dc6 |
| graph_prior_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-dea77af43d30a511861660f5f8d2b76ccafba0d0fc0537b5de66ca701fadaaab |
| learned_route | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 2 | sha256-04b4d4950a94d662853fd6128aad724ed62b659cf792b18372956a14ad653c5a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-fbda24cb | sha256-6ae7974564a9e5f7b7c0b974bf34595c0946652bd1f7e775ca168079c7b2a8ba |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-fbda24cb | sha256-1be9288809056365bab333db38d7ed474696bb4fbf601bb75e8c61c0aa74b99d |
| learned_route | turn-1 | 70 | yes | 1/2 | no | no | pack-fbda24cb | sha256-5e9dbc183f9ce5b9d6b86682be4fdfa1ee14000dceae6ba43b81a8155ed7dc3b |
