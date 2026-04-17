# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1e91f891be11ad983e343a9bbb8eb7e094a3203fdeb0cba32d80844dcceadc5b`
- fixture hash: `sha256-c962d7bf59f91132e81f529b35b43a46128d3cc144f19a803783e383eb2588e0`
- score hash: `sha256-06d104370caa29e7aefbc88416976bc708ebe202c21f03ea5fb31fdfcd7f4394`
- bundle hash: `sha256-d349c5de0f685200bdee819d6c6b32a52dd4bfae3386592c2838f6937d5ea908`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5abc89ba1c4aafac24d8b492241ea58c50f7925494e6166e3016c9a753e61584 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5eddf652f7a650f2bfcf25ba2c4803f356f502d261a97b4e019cdb5243566739 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-15622da0d05b0defdabeb591d5996b608a0002710ee26390a6a999d05a52b5fb |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-df5be748752fc046cb22b4437edbd0c33349d423b54fce2609ead3ceaa2507be |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8e052775 | sha256-ad360ba51d27cf4a8d7a20c0e8858ea8911ffd279c54e0a1c3ba590ff28091f6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8e052775 | sha256-5368ddbd7f475d19d0ab83b3beae5a05cd39805143dd588cd7adcb6d2f1a6358 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-997d570a | sha256-01f61c145c7d42cfc88e1dc9ed9d9d4d7a6e9f0ee7d4b8d25b1d2cb5c3853406 |
