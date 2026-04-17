# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1c2fee7fd4eb0c2720a3ba15050df8108cb036feca9a01fcd35c4b07aae7a9f5`
- fixture hash: `sha256-61f4419f55eaa7d0c0ca68a6f768711b70a4823f4e0fe058cff8927193ee8afc`
- score hash: `sha256-73c36caa30c0c70079f53f4243f2b1e9f2fbef53e9ae75d3365b1376c6e0f9b8`
- bundle hash: `sha256-1544db054e9fccc311a548200c8a034f64dfe3433b59b0ba747f3c64bc5d0020`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3dcc36ea1001cff13b10454b28af88c47e797eba5193d74b4990d61c1caa8eeb |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-d33dcfefb2ab49e992ea9d3e891a3469b4565c50007d5665c186587cbddb65d0 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-709e207f5aab6854434d5fe7e1e392d0a2c7be29be063d9a2ed6599ba6eeebd2 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-ee8f1abd9dc26063ab335d1c2315847dcdb0e4c0af68af7c6baebaa4df19166c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d915cae7 | sha256-af8eac66e33ed6892c2db24115e730dcc04cfdaa9034db16949fd6104922e008 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d915cae7 | sha256-c57c364acede7d1054cd37a56a1b217e2bc0dadf97b71dc7e9d45cfbc1f82cd5 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-d00edfe4 | sha256-0dde21ebd1320a9c7c08e7afa7ed7a3d44260c7d86cf0ddac392be83c3f5ccb6 |
