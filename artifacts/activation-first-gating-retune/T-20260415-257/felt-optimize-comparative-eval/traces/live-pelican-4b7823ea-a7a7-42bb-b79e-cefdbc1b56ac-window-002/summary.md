# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dfa81f4d2b00217c5c5c520178573740e8780c6997e7fbd463fe714331cc7869`
- fixture hash: `sha256-ed00dcfbed6598ace12042db40479b3199c9a2955a7a673a786b8d8fa048ed17`
- score hash: `sha256-1b7263632613c5446ccd40cd5247b852f5032da2ac6733ea04efbff0f8e651db`
- bundle hash: `sha256-002d6f23f0b32b18cf06a9afe82a1e79c179317fbaf47cf7c9fc4f2239064f1a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2d5028651743004fc65c4abb7a18a3ce781f93f13bd67703dbd698c51e61ae2 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2d344b490d07150a21db5f614d6a1b36820a187cf95f019ef50d0cc48086c09a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cea54e0cb0d0d4edf2f88e85676b81ed0d9859a613ab18bc1c8123139be5bf85 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e8ec8da4db8b8ddf61d5ee3df64932d07a0e009469f9e0008a6b11987bcd3065 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-acbe0832 | sha256-a281c03037ad9123c94017f7825402da971741562b3be699f177087c9b3162a1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-acbe0832 | sha256-a281c03037ad9123c94017f7825402da971741562b3be699f177087c9b3162a1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7ef55951 | sha256-e355cee846ba2435b04f81cf0a5601f2258b39e6cdb87c31d46f2287719b5f4b |
