# Recorded Session Replay Proof Bundle

- trace id: `trace-plan-lane-handoff`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fb2dfcc53391da243d2eb309363317b16592b228eb91cec9df07b31f719e0068`
- fixture hash: `sha256-f1ba04aa90e022e7a4283505f4deef91fe6f05bb8b561d462b99f13bd3455652`
- score hash: `sha256-873fa430060dda5c070f764f103e32e4fca215234a61be2d9d3b75fb9d6c79c6`
- bundle hash: `sha256-604f95c1e7d05312384db461aa9fc3d46309e7d174195bfd62223a184bed1d9b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 15/20
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/5 | 0 | 0 | 2 | 1 | 0 | sha256-91e1ef9d2b3d61940bcc1c02b134971c7b50bc3f88ccf5323436d9f8cd822fe0 |
| vector_only | 2 | 2 | 5/5 | 0 | 0 | 2 | 1 | 0 | sha256-89aad185395ef7316a588a86e73bbe14243e418164743c20145f3a974eed3a38 |
| graph_prior_only | 2 | 2 | 5/5 | 0 | 0 | 2 | 1 | 0 | sha256-a01c61ee34abd7a01717c581d122a78577625cc375ea2b059a1323504ceeb028 |
| learned_route | 2 | 2 | 5/5 | 1 | 1 | 2 | 1 | 0 | sha256-6d3282f127b3f7d3ed2c2a31c526868f525266473f9dc2bd57beed8f843dcc48 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | lane-handoff-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | lane-handoff-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | lane-handoff-turn-1 | 100 | yes | 2/2 | no | no | pack-cd0948c8 | sha256-99782b96c0ec35bf8deefda910ba61966ccd923a4eb3ff157d1c3788366e06c9 |
| vector_only | lane-handoff-turn-2 | 100 | yes | 3/3 | no | no | pack-cd0948c8 | sha256-99782b96c0ec35bf8deefda910ba61966ccd923a4eb3ff157d1c3788366e06c9 |
| graph_prior_only | lane-handoff-turn-1 | 100 | yes | 2/2 | no | no | pack-cd0948c8 | sha256-99782b96c0ec35bf8deefda910ba61966ccd923a4eb3ff157d1c3788366e06c9 |
| graph_prior_only | lane-handoff-turn-2 | 100 | yes | 3/3 | no | no | pack-cd0948c8 | sha256-99782b96c0ec35bf8deefda910ba61966ccd923a4eb3ff157d1c3788366e06c9 |
| learned_route | lane-handoff-turn-1 | 100 | yes | 2/2 | no | yes | pack-cd0948c8 | sha256-99782b96c0ec35bf8deefda910ba61966ccd923a4eb3ff157d1c3788366e06c9 |
| learned_route | lane-handoff-turn-2 | 100 | yes | 3/3 | yes | no | pack-252cb251 | sha256-3f51eacc08f92d486af36252f8d0e18fb7fc1aa7cd211597ebbd1cfd3794372f |
