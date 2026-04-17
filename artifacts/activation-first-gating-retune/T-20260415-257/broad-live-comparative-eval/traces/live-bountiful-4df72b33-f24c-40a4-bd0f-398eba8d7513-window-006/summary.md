# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1c2fee7fd4eb0c2720a3ba15050df8108cb036feca9a01fcd35c4b07aae7a9f5`
- fixture hash: `sha256-61f4419f55eaa7d0c0ca68a6f768711b70a4823f4e0fe058cff8927193ee8afc`
- score hash: `sha256-8b92a08268a5422ab675491956a6ae089720e97b9400a414d15f08226d528165`
- bundle hash: `sha256-82a86f743b20b9863393352784f247a31cdb02b662866258ce52dd0926cc3195`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3dcc36ea1001cff13b10454b28af88c47e797eba5193d74b4990d61c1caa8eeb |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b9514e9094f2b327be9136f134fcddf881cd136356f6207768b6427322260490 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c8dabe4680fde1aab0950da3cb73a1e92bb99c2c7f7dad3124db06e338395874 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-00163dbbc545c294aaec6479e2d97fb0a341cb1a63d1a06d41968d87b0cfc68f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-15951f89 | sha256-01e65b6a846ace880598682c2e6cef5db0073f5e1f9c7ee8795056af74caa853 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-15951f89 | sha256-a3ac4a4d00e66761b3532b7b4d3b3ddc1494ef81430b6685bde44f97bbb82c86 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0c8e3486 | sha256-dc3f9bcdcd21ef273fb443036c5f51f30533ea7035ddb9fefc670f52233ed45f |
