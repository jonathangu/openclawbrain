# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60359ed78d5b78e9d115bf8cb9e9ba270e0f90bac409bf6884d4a443b2440f94`
- fixture hash: `sha256-0da91a494c8a34b6c27eb293958b781dbe6bc334337372f9fbd368fd3d0ee08d`
- score hash: `sha256-d775e0876c111f1d20d8cf3cfef4971c59e139df47ce3a7cb44dd29028189c4f`
- bundle hash: `sha256-59fd040a59231292ce8e3b0beb774ae5f4fe3ddb44affafc5753e2258da93720`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c77331726a9326f500ec3f7c3dbbaeae387d368e17255232ecaec7597f897fed |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b2bdb8819ba05a342b98b9b7060d9ee07dc7f12554556b1d39ebe5c61dc58d25 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a886125586bfa38b8036ddc5b95b23e2e4e111bb32fddbb5fda96ae03fb36a28 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c514b99db0e206b9f3e7c73012c81fd13e34c53d871ca375228bada45ee047b7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-146f78cf | sha256-a1e1d9034f95bcfe2e9a1d02b6cd19c514dd2d22bd843da848fecae4c5d45e81 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-146f78cf | sha256-6c2b85e45e1669f2663966cc6ca416c66698c82ca96a01e8fa012f01516e386e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-294598e8 | sha256-48624f0e7cdb5671191f66a67796145e1a88e29097e50cdb797b2807dc0d32e4 |
