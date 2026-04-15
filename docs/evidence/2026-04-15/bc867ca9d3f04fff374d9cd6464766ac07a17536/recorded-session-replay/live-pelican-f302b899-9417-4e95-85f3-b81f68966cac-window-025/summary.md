# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1e91f891be11ad983e343a9bbb8eb7e094a3203fdeb0cba32d80844dcceadc5b`
- fixture hash: `sha256-c962d7bf59f91132e81f529b35b43a46128d3cc144f19a803783e383eb2588e0`
- score hash: `sha256-21f72807a96f5ca2297826768dbc7f4e3a0082410ce217039c2164ec0830121b`
- bundle hash: `sha256-ec1abc256662760bd6b5ca9b23e629e4d57a185930536a674c196fa5b80385de`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-352546c474d7f8b499cdfc06757848306589638c494ed3b06902efcdf0ae9476 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1037183cc8c68673980229c3b9d06e017813d818ee38d2edc15a5d87c28a1d76 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3c2f1074f9387cc837d6a3ab721570f84edaad31281433273d50c0d26d17e0ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-16b4e1eb | sha256-f02f2f224fb0811f0bce23f36784911e63f5276ef121aa24797a8368334e1bc1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-16b4e1eb | sha256-9253e1c88787a0a4cfac0c23752a7a82e2c548ad92e23de9c9b5c8d7656a6aa7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-16b4e1eb | sha256-f02f2f224fb0811f0bce23f36784911e63f5276ef121aa24797a8368334e1bc1 |
