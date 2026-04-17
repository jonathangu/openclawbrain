# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b9d58238866be4c30cb67001ed41c476bb074abc457ce91f27bdf2a95087dda`
- fixture hash: `sha256-93a191f41c9134f7fb1b39f4120c598d79722f0fdf720a1c60726eeea45f85a7`
- score hash: `sha256-0873679a0a4b87b70a8fd1d437a064b8ffdd714a0df9b54d38e4d06685f1f3f9`
- bundle hash: `sha256-cc68456247e3ef7e6cbd038a6403c9473f3c45021530d011b0e3978375c51f3f`

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
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-53ac170bbdfe31610a82a7fea6a20f739ad327e9856e23aa713b46f86601ea52 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-a50c1f6f1f8493a189b351db513bd7f4bfe65f284812b5bbeaad2e5c6da09cca |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-2094a489d37d16d27e45a03bf95e4e1f48ae2d2eda93174876c571e2622d9a18 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-0e7ab81aa3fe427bcb3ed7d42bd4d09a92bd8728ecd8c89bb2c8923358cf0b24 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9b2a810f | sha256-9d03a37d0269db6d858cc72b8e4470443a815de044f40330499eeb6232ce6234 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9b2a810f | sha256-2b44469ba6704b449b77e892228b6da7c0a84167083254b5657d6b00dce48e5b |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-79a24fec | sha256-4de3f0c0dfd04fa0c70cdc7180ec1f1219d9a7dd2f11c819931f8357c335b6c2 |
