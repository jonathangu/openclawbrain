# Recorded Session Replay Proof Bundle

- trace id: `live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4d327d918fbcc4a369abe7ef4d164f9a5cfa04faf7adb2505c432935e6de9ae6`
- fixture hash: `sha256-6153c43221a8c0bd8b8f42dc9046e70b2d1a03e5bc667d5e9fc62b4aa1f0fcb9`
- score hash: `sha256-20391dea811d3fb4dafac985e3f198156c7922d7024431b04af00277957177df`
- bundle hash: `sha256-dbbe50e42211738f38529d57a802895c1688da06cbacc2e06a9dcf8780f27438`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d542a8f0c800204dee4f72d85787e8cb1b923c865594f3befa573eb5cd2d9388 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-fbbbec05e53e90b09d4ec168dafede9d9c5f6f175652ca0757c4e5366fa5d964 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-59d04fbbee264cd82a710638474ba769375f4f6cf9ccceab2684a62080c1c2a9 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c462343be94c34987878f33b667da8e51f29074c0373b862f842227789ea3df7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-6ef8d51c | sha256-c96c97b693f4e3fe881c3ab038a32c904e50c8326e73fbbbe517bfe199975095 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-6ef8d51c | sha256-3fd854d1b0387bf4373c37737b0c530a29551c35dd63de7fd95f6b3e4941168f |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-6ef8d51c | sha256-4023c14b85ce93cab9f6fd355d3adec8f76ab0b471a985bfac344a3b571dbc02 |
