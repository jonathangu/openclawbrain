# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053`
- winner mode: `graph_prior_only`
- trace hash: `sha256-90fa2d4d4139a1ebe44459c22c8727829266007877108f45fae96bd38c29ee19`
- fixture hash: `sha256-cdac1a3c1c52fe7f99167e3c99b2296e6c3d58fa57db9f5982bd144cf8ae1b02`
- score hash: `sha256-20ac59c08652296da52c1dc726708d447327d591f92b37905f4b95ecf91b9a3c`
- bundle hash: `sha256-94c7720a8dcf1df5c8f913f7613ec4c1e7620d5c93623e80eea931d6c57ecc0d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-24482bbabb11a52346eb943534814017cb04b7bc645577c10d973e3de61757df |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-84c7e77bb7ed9e1f166c82e714de772cffaa13dbc107e057fdb041a51b0d9546 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5a68547b38eae9831ce7526bd59e161587776b00011505c03b00d92279a8f0ff |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d605f08c7b03eb6e772612be5900b7ae9dc3b836938bf7eeb573725204da39f9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-712f47f4 | sha256-2dbcaf7c4e8f5d945f4f95f5bd3a1dc7b99f68e08afcdf8fdbccd1b67efd8cfc |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-712f47f4 | sha256-ac7e8eafdd4d432f9459cc8ff960a8628655ba0f705ba3083b8806acf12c9bbd |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-67d14db1 | sha256-dc952a975143af85767ae06899018d4d3495c8f52e987c156c9923eb7933d735 |
