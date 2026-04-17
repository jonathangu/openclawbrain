# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7768c82b82cfc9e79b11d2862950b229d848ebccb180ebcb1860140cf56b1f18`
- fixture hash: `sha256-e3d4346656fea9fcd52a8093d89ccf43c79e719fec02594aace8851b57c7f190`
- score hash: `sha256-2ee40b3cd044b49074e9a287f724a12ab70a3fdc78894a5591be621c805e3343`
- bundle hash: `sha256-922cf171b3301b2d50594e6b6203c534c248f3cdf47eae90ee46609fca761a1e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb4473107ca7b170cb7198e9a132dfc26d383b8a4567d404be160b76d2d08390 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-743def59275f6986fb6db2e27d17d28689b911a130f69dc695ef3b67255e25fe |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1feba5f422e6f112ad5700dc6e2f7577fea6faecae9062c101c223806e235a4b |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-80f4ff0534ab7635616f34f23a0c17f98f09c808fa4fc51ea1fb900900748f2a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-05e8df40 | sha256-6a902a4351efa5352a46aa992c3ffccd9acd30100548383b9fa751b03af70679 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-05e8df40 | sha256-2d772802c455db33d3a6d6af93d80164385d77af76c6ea903ba7abd66ec7dc92 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7fe5a433 | sha256-5c610887234836740e88c0b06786ef4588d03a73e721ad344b4bac648d22172a |
