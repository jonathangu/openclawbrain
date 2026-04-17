# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80f4bd70b8f229336838d17a92921bfca64745f162a9177679361b11e355a256`
- fixture hash: `sha256-1a25c630a19d83ff4b3784d9a97b879e228a965826872ffe6bcc2e6453fbac5a`
- score hash: `sha256-ef63b52ae5c28c6d7eb444bd3859ce2200e11f1372a6313c7d7b3571b4bbcdb6`
- bundle hash: `sha256-bbb77b9f19263f85c6b5e5d574b53b7c4ceaa19214e391bfe5d11f518633bd33`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94ade431a5c986254405c71afb4d4071b897c04f7cbfe57133fcaa9500ad06d1 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b28d2e5305b29da5a6994e7e48aa5391df45a64ac01e854dda3c73dc369cbeb |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e354cc35d8515c4c442641441823d181a4599bcf4c56ac87570d672403dd32de |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6c6bbcc0e355100efd6906b582fc4ec70ded4f8c89dde32e8d320610a8e03fa9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fee82a0c | sha256-9267326a5a30d63bbf2e1dcc58071670e67fa976b40ddb7b453d86eef08671be |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fee82a0c | sha256-c86df4100e9736060db6cec1fc5e79f7ea944792dcd72fc11f07594d217f1f0a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-2ec1e45f | sha256-ac0444a5377da90dd98199189e532257a90503974289fa32a826c3fdfb0098e7 |
