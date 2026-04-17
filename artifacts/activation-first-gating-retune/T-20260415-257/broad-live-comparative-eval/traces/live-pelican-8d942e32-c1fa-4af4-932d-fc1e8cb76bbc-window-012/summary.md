# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-874a83098560adaa94c38c7c63cbf4c86efe4c86090d606bbfa34849e336a8c9`
- fixture hash: `sha256-b06776d862580d01d558132918aaffc22b9130c1387f99ca2438e1c6cbf7e22c`
- score hash: `sha256-2704a72775f76e7142ae8d28e55ec86c87a5cf875c1c0a5715a813007c7fc78a`
- bundle hash: `sha256-26d56e6aee08cd3d3b22294c2e4a48a115daa0b022bb405b23d91418aecc1da0`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-74c569833e5bf27fad2f2f842fa8eaa7d60bb320f690bc493bbf6c394f309f6d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7a7afdcbe0885672f93dd4cdb8d242368242e235cd1a4c9b357ba903b5bf297a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1bdb80fdaa27afda05791a4c648e2aacc00e4f9aa6445775dfc5e5501c48f870 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-61d0f996b406ab36bc5fe22abc1d9e9204fc8087bd85e2df94cb4e74a91b5ad1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ed8e2d91 | sha256-1fe1e544bd188536caca0acd72ae2153500e076c3ff751c5fc3d6be11fa57912 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ed8e2d91 | sha256-8d864c8751d38ccfa8f49d04fa1154428fb5a7c2fdeec81ef04ac5459cd6558f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-463f9d92 | sha256-ac1b79b8b97feede4c8d8d6ec8ad7f3d6607935dd83440a5ca5acec99561abbc |
