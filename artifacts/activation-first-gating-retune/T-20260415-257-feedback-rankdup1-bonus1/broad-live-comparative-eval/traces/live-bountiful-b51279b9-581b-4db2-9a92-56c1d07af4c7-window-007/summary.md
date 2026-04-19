# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-791d7ab7e1a39f9248affb0ec38376441778feefca919fa6d5dc852b64c0c740`
- fixture hash: `sha256-fc8733ed1be81b69ef5447410e17ee0e67ec342cb6d0c7a27eab065d2955bafe`
- score hash: `sha256-1c97a3bd5fbda70fc62744a570d73497a61749a552aa4c5928b1bbcce287b41d`
- bundle hash: `sha256-04e5a1586893feb4141bb8c8a182f13020d53c6d7a056e6405ca5189c48883ac`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec63dafa723fc5eeeb737d59bb6d87f1a6423a6bfa624fac5dd61b64e8a7a79b |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-938bba4f5769c0b6cf04be3c2c902aea64c4c7073a9d534346b263d04554838f |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-3521f47c22ba0198a7f0e126c9a00633111ac22b0518bae31964ee0738c654ec |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-22eee53012a38d545547e40b272adc3625c64297697c389e33625e8a67930a87 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-464f5d9b | sha256-d5427cd9919073c2374da219a798a7ef021c4311342b9e26773eb451187ed463 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-464f5d9b | sha256-b588ce1e31a42f147ed76618425e7b0482db44ecfdffe05141cd49227e91a5d6 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-464f5d9b | sha256-f0ba9645094429371838d08b9722ac9544f45cd83271727d28c17ae30547fb28 |
