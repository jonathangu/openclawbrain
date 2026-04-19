# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-caf66a8ae0f50cd2a6d3d3cfd5e877c0e6e80108a88f78a99bf9d0af8e14c2ff`
- fixture hash: `sha256-d0e5149f0ea8ad48a690b98cf321460b1ba9a083ea4dc63f8728a3baa728b105`
- score hash: `sha256-319b718a3a987ff34a5904d8ea3bbc83f0ffbaaa33829833e6c06b0100ab44d1`
- bundle hash: `sha256-8acb9d000b2ffe7dac5ab66e074161dbae4c4bbeda67708cae1410d10e87d8b3`

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
| vector_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-af66721279054e76e8676cb6e6ba6b34c05f5ebf6aea36ddbb161a78d8531771 |
| graph_prior_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-2e0fa634de4c7334d1296b20170d7c67088a4d261744f2a5c5dfed51ff95dbe3 |
| learned_route | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 2 | sha256-2ffdec345c7e68a08c46db509acb28f1f244186a62ba46e9b199558bbc9ebfb2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-4197ecd4 | sha256-24c002e2e88880c121a5f7f5fcd7c461ce9506f507c0bbe92fe5db8cab1d9f46 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-4197ecd4 | sha256-8ce28ca866af04ef32a2ad063a7bdf34c10cba44168830dc9255ff4676f7e9e2 |
| learned_route | turn-1 | 70 | yes | 1/2 | no | no | pack-4197ecd4 | sha256-f9ead01f608d16e9da8d4343cf192aa34442f1243dc711710a584ffc90fce3e7 |
