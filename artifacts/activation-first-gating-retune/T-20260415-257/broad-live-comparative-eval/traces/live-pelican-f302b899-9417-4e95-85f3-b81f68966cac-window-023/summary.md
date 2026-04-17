# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-858b0f43ef470e5eca2afe1da13983b8601f538afa969d1fec1c6c995e06b43b`
- fixture hash: `sha256-5c10bbce6c643d206da3406e24da637074a598036c8915b8615377d8cba78cd2`
- score hash: `sha256-ec26cdb820d5a46929008d2ea2c8ed6a208929a3a7fd5429583ca956dd5801b4`
- bundle hash: `sha256-d7722348d2c3f3565c84a7cf6841b541d9b5484dddec9c585375cbfeaf20f1e5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-17f51076699069dddd3d95709feed654148e93f103c223926ea5c4a4e2da537b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e7e4bc6083cea9639fe7ed42e7d97dc4aa9972b4f9b867693fdb7a65b8bf77a0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0980567f39f672f2d4b37f4e9fe40dde658e545b69ab671c9c7f85420f21587a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2afad67c7324f9959fd7b6ffc0525aeec7a1c4c8ed949380fc976eb18e10bb0b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-60c25355 | sha256-d2b99f3bacb0858ce43f86e588369513dcee82892c00e50ae6f888d737d59e7f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-60c25355 | sha256-ecdabdc37eebcb25f6e5739a88d4034af42e159e9ff50d034395936e80e90dd2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-215ef02a | sha256-91a3e1cbe6897f797f5e39a6a003b474dbbbdd855a80ca70dcc6f6578d2c0608 |
