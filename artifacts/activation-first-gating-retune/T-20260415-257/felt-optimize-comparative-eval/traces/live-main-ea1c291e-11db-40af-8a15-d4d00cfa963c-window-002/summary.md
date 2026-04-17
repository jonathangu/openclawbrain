# Recorded Session Replay Proof Bundle

- trace id: `live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d851cdfc065d530ff6a05cd12aae1453cc6c5cc252f286f05c63b39f7b7ea103`
- fixture hash: `sha256-add4e01555ea0b700f89e1179ee076e863d3216d180ce57f607f066d853c468e`
- score hash: `sha256-1b532a122d6f91f4265dcbb45a3a58ce883cc41e7c7cd71621a9b71812d2da9f`
- bundle hash: `sha256-fbcdc091887084a997e5844776dd36b3f4e2e7fbd9d3e26c21f3f360ed2cd884`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e16f2c51fd8866c40ce249b661c20fa44d3a586d3c45a550284b22e35e90bd83 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-801df0cf239f321ace430035a98bdb17e273d35bf8a707145158058109a24430 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-65b8b4cccc5f4a748b1986f33bfbe8b034d596cb1bf2b28fa0daa72e0339f93e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-acca6bf6ad1d2986b8575c0b0dd80edb05a377023079dd1255d08703ad4f4739 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2409b85b | sha256-49fa66a7550524dae51b066014804edd76237477829efee14c7c1407081b22b5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2409b85b | sha256-c0356be3701591a1bfee516336fd8dd8ab9a74c43136ce22d05122c6e300951c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-953cdc58 | sha256-5371281c051cc6977b1fd5fb376542c6b2a5d6cd4ca6382af6350689c40a6a42 |
