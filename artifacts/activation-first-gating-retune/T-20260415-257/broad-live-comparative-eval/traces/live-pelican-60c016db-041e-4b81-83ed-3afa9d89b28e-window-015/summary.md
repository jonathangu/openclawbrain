# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-19cd6a701f3afe5404567d59955346d7cfc26c77deb7b29e61fccacc22d3bbfa`
- fixture hash: `sha256-4dda7357e5652f879faf39fc4f606d23e6674326c96ea6b533ba27ecfc72cf16`
- score hash: `sha256-24be08d0d9b73742879d5c6902f477c83b340f5cf655b23587bfd8a9541fa500`
- bundle hash: `sha256-098ec58f5b83a98cef0d64ecbe3be8819ca546b2aadb6343636ee227e021e53a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-934729bde748377658ef5251e3c9784137a24d5cc133cff448c2ec475fa6a4b7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d678efaa44bfff426025432f121ad4ed1954be7b2127b02807bc27a0d0614e63 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-84e66b316a75ad5f6abc4824ded3d105064aeac5ee0817e590e2a0b81b6851b9 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7b86fd0a6349055acb0748800f97c616e4f9ba0125e8fa3160143f163a7245e2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e15b4ffc | sha256-593e862681f6dfe5c8ea2c790f214813379a190c80e8fabfbac5fef7ab09c174 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e15b4ffc | sha256-f229c16cb728c82f2e51d27b37b682058442532684fdf7c0eda99c67122135df |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-c4d6ad5b | sha256-56d436f80798c94f533c380741e8c7410d76a0e11cbd987330339d226cecdeda |
