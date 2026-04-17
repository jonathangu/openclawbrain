# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-347a801443da9e3d23f8dc976f3d286dbcc3cafa0984aebf1f93ff8efbfd1773`
- fixture hash: `sha256-3e9f54e7049625692dd39972563612e44cc8adf4a2a27dc80d450c5621a5caf7`
- score hash: `sha256-519eb4ea3fcf1196a335976d64d994567914851e23fb60a5de08438fbbc1bfdf`
- bundle hash: `sha256-497034cd9b34acc503fa32f1a5296a0caa4ec8c7fd9cfc7acfb4c73a0e4d05d8`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bc058e6f191036e6bf4f3884982c6a502fc3d927441bbbd1c5d745ba4e254aee |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5633202168a52c86d3c15ae33dbfd9daf5f84a8c3cfbb048b902f6bec13939ec |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7f9d97e814183a6e54b92d6715bea6020e49f9ad9b36ae62af24db3d9afd6eb3 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f59afcefaea4c44dbd9433ee000f3cb6727f64a68a45cca26e5491cfa5d3e8d4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bf491d0a | sha256-6a934a2eea60b66232c810ddad1b8e6951eb91017b24f08eb3987a0e358992f4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bf491d0a | sha256-6a934a2eea60b66232c810ddad1b8e6951eb91017b24f08eb3987a0e358992f4 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-7a9e9a37 | sha256-8a7ac6ecdcac3a96ce14ae261dbd7e73a0a1c385cbaceb9800ec6fc9c434c94c |
