# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b6b834033c55b0c1fa4b1b699a1d8df0de8cfc6135b0ef6c168d6806431da077`
- fixture hash: `sha256-da3f651615d0891ce7f18953e9f938d1c01935c73ef17cad6fec24cf102a80c5`
- score hash: `sha256-adb8302938a8acc6b127261460d477d9797d1e58d68eebc399f95cb5d6aa6f44`
- bundle hash: `sha256-1b6d1144769c210706718a8cca63c9e71f72c12af7b7b352c4228ce00a98758b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7bb32e7c3cacde4e347cf09cbdabeddff358a8d0a73f0b6ff3c688033dc4ad3f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f070859d6c2d2f5439d23645a8ff18c8aada6e52681eeb3fbe4d2c9f67f7105 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-83ddd153d6e5b63160d71724b5f28455dd8054f3d8386dc1352e0d32388d3f70 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-817da14575b7f192911fd31804ab17771fba883e41ebc5c703d3b7ebbb80b4cf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1c7c36bd | sha256-0695f475e942eff1f9164ef3fef3efaebf2bf79378febecfeb7f4ca25749d0b2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1c7c36bd | sha256-0695f475e942eff1f9164ef3fef3efaebf2bf79378febecfeb7f4ca25749d0b2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-01222d60 | sha256-afaaae4563ad4a4edfc8a4218c12d74375d6e7887ef1f2ddbfff3478e1bef508 |
