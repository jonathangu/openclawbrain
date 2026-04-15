# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050`
- winner mode: `graph_prior_only`
- trace hash: `sha256-edca99fed5bfa02f8196eb002c7ca5449e3ad77a55a7ede4db613e0216b9a288`
- fixture hash: `sha256-075e6ae40abca95623d0eaab9386a47facdde038c6af2c88f5255ae9a6184b2a`
- score hash: `sha256-6a95e0e55c186e4c7ae56ddb9d936cd31e3e1f0a2d461511cb599de8e07bcbd2`
- bundle hash: `sha256-c7dc09f4f37e42eac560f21ccda4c5715143389efb2ab59c57000169aa0c8850`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6022525cea9be81df298416f8768122250fedda0d93e17bc4857c9bee2c2bbc7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4aac168924e188397856a8449513a721df9964033040edb496cf326409289c23 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7a3f6006725a4b308101eff2c2f2515d4acab9ccd4f259b4f4aea314d036169c |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-632c08e774d9f7103a0dc9ee6ffddb351c5e0e980c87ba20f971f0644fac89d9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-22c2f7f1 | sha256-49ac13a9c0fabf1fbb577ec5b9875a7c79f4fbc26b9d2c1d341e3a9b1ceaa4a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-22c2f7f1 | sha256-49ac13a9c0fabf1fbb577ec5b9875a7c79f4fbc26b9d2c1d341e3a9b1ceaa4a8 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-22c2f7f1 | sha256-49ac13a9c0fabf1fbb577ec5b9875a7c79f4fbc26b9d2c1d341e3a9b1ceaa4a8 |
