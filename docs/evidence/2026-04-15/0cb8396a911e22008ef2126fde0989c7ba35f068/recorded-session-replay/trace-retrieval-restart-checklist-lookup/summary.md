# Recorded Session Replay Proof Bundle

- trace id: `trace-retrieval-restart-checklist-lookup`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6f5f930f2df03ad616e08dfd2f0d1e71d1aa99ff5985e8434e9947ee2ee26b92`
- fixture hash: `sha256-ee04d0d989fc57e44664473fe8d656d4383122c7b35b3a2ebbcac2b5c1400aa0`
- score hash: `sha256-0fc01587af55950c910cdd07e471c83c56df7f9e85115e0222b4f69990284a9c`
- bundle hash: `sha256-9fe2290407bf848483ff28aafc903f735d4e94905561b8e6d3d1a69a1efeb16c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

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
| no_brain | 2 | 0 | 0/3 | 0 | 0 | 2 | 1 | 0 | sha256-d4a1b84ad22f95796c1566af294f5d58d89f7c13c9b22da34e8c8cc232f5be7b |
| vector_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-e2d8c9f774eb27c82abef8fbb45fe90866602243771f1a3275caa70d17003858 |
| graph_prior_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-92dbc9a9fb131ddcd42ffafdf13c006d0d8a09c04790bebd1b570215c979bac8 |
| learned_route | 2 | 2 | 3/3 | 1 | 1 | 2 | 1 | 0 | sha256-fabb3e79336072a2c9259f6b22fc379994f5a599fe24094017c4a248417d813e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | restart-checklist-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | restart-checklist-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | restart-checklist-turn-1 | 100 | yes | 1/1 | no | no | pack-83ea263d | sha256-b66b5a0d85c95a815f4eb21c09860ad80d4e78415f0292d62a2101fb45e11e02 |
| vector_only | restart-checklist-turn-2 | 100 | yes | 2/2 | no | no | pack-83ea263d | sha256-b6a5758d54f00fa563a6b25550676ef2ba3685a52f6a04713dff8a0fec755504 |
| graph_prior_only | restart-checklist-turn-1 | 100 | yes | 1/1 | no | no | pack-83ea263d | sha256-b66b5a0d85c95a815f4eb21c09860ad80d4e78415f0292d62a2101fb45e11e02 |
| graph_prior_only | restart-checklist-turn-2 | 100 | yes | 2/2 | no | no | pack-83ea263d | sha256-b6a5758d54f00fa563a6b25550676ef2ba3685a52f6a04713dff8a0fec755504 |
| learned_route | restart-checklist-turn-1 | 100 | yes | 1/1 | no | yes | pack-83ea263d | sha256-b66b5a0d85c95a815f4eb21c09860ad80d4e78415f0292d62a2101fb45e11e02 |
| learned_route | restart-checklist-turn-2 | 100 | yes | 2/2 | yes | no | pack-4affc92d | sha256-8fb57e5413af310bb2bbbe58bf3f0b294e4c7bc6203dc39bd0691c8c26521220 |
