# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175`
- winner mode: `graph_prior_only`
- trace hash: `sha256-653e1762b7192e93df1dc01ad3fa2126f6513bad2e3d5a89891f193ded446910`
- fixture hash: `sha256-4262eb1c667bd83d27b33dceb3d4d1a1c6a1b57d1ba763770502ff6e7c8a4239`
- score hash: `sha256-3969b8cd608a838cce0ad345782dbb348bf10f1225b704ee9bce2970731992c4`
- bundle hash: `sha256-d3d1988a24276b49bf7d6e2764a547a4aa49315e121a28634b5cd267f4fd15c0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f90cb77dd3ff507938d3ef155b0e74e6914215ea5bf7fbd610cc02d8404add3 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-37f431afdff2c9de6c00ef4e4a56b3908d3dbdf359e91bc22f041bdc5fc50e82 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-061425d78ce1605d0d08f117769efc4ef65dd371b9bc0c201a184a15dfd798c2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-62177a96015bd160bff13ce627e10fa03b8211b59a30658381999c9c56180de4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-030d0936 | sha256-90d2e9b19be93442c89bb6fa410c900e1cf730c73f681b48d574779c5355016d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-030d0936 | sha256-a83818522976b0a4b8ddf14496567a944c7f81e262cf5443bd93d4233536a197 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-dd3687eb | sha256-7b539ded58143a212bc352ae0e6a117763a8cbae45f7112a7e19093aa201bd1c |
