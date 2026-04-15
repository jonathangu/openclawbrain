# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b973c64c8eac5a0b6deba25fbae9f31be4599e3d19192c7c9dd0b18e718f1e`
- fixture hash: `sha256-b932d5e627b7081f980ab111b252e205aa7e0185bfcd774e6388fb9e948098c1`
- score hash: `sha256-3c9dbb482dd427d11eef15b0f58b6e5019b15d943c3d8bcc8211c2135a097389`
- bundle hash: `sha256-9c589aee9bdb0854fbe8e0f7df70b5d284d9ef3b61428b18c9a60ce9ceec4dbb`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fd3c28bfccf2817f3d01d14dc16c97875abfde806e8cfbeff2d04b6e2a397e7b |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d34c578fa59f0f74c778d68c7ddde0146792fe3fbb173876c3f6bec52e801848 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-38272cce2aac63cc3c6f1822a216b31935e1fbb3d57dabaf32545ea7d77f3361 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-511c68539a1a45ea30a53738a025dde22654583dc155c88e4ae735818cdfa372 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9025dc74 | sha256-9582f408a574f7b9463be563f0276b2efa9c102d1004445b65af16cd4886c35d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9025dc74 | sha256-9582f408a574f7b9463be563f0276b2efa9c102d1004445b65af16cd4886c35d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-9025dc74 | sha256-9582f408a574f7b9463be563f0276b2efa9c102d1004445b65af16cd4886c35d |
