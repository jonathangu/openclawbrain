# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8836c19286d7dfc7e25365f74f5e0786007d5f48b08d8fcfba5fe79b0f03c2c`
- fixture hash: `sha256-0ffaff36365448396a5594a68d8364ec6eacdae9fdbcb2693a4ddbea65547f4c`
- score hash: `sha256-00ab279004c60d7609b6c51eeb2d248c80b638d4fb98c142445af1b25ad19616`
- bundle hash: `sha256-b926fbe247b1becc0ee7b8e77dea7f9a34b08cb80b847a790267369af55bab84`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6c4e67449219060e0eaa53a64e9ca0f2f7168ec707e126564ccb072cf633b7d0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e47ee8a1790ae956b0e6e3c341bcc3459ef576d5a38a22c18ad8f16efbbd7d4e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-38618aab6b7a532cbc71ffaca04d6d36fec7dcb3d86e4bac5f73e14416b7a7cb |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-4d489618130285b5f062b697fe7bfec49e34d05e559091d92455919e362b62aa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8508c2ef | sha256-ad6af4c81643a77a9481df142a31acae61aba2577778801a453bc85d9fbb1716 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8508c2ef | sha256-a712d21f8ee981a6e20dc690de230defd3c0b22c96aa3f52767386d9b10c95c1 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8508c2ef | sha256-ad6af4c81643a77a9481df142a31acae61aba2577778801a453bc85d9fbb1716 |
