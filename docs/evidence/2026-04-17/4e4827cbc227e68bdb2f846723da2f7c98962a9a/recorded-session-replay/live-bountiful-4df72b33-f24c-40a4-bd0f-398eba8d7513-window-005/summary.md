# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3abef879206da47064eddd47e25da3ed69b90db7cd3c4a8ad4966415b7f00bfc`
- fixture hash: `sha256-9918ac1f02e6942937a0c165ef4e1221b4c237d331f00ffb8e89f19fa2868433`
- score hash: `sha256-465339900a5047c5defe40b1c9e6382f2b887124f987429aed0af8bd4b667d80`
- bundle hash: `sha256-f87f894d1c4ab4eb5e670ba49cf368f1939d8db1d52323bd8f59ae97f4eb800e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29b14ff43615dac430701017e1a95d84a605d40df7e69393e02bc78849368384 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-8d19c5f9e17b8d7266fe97cf44eb97917ca55e7c10a910f785d62ad79abda24a |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-72d5ed5c730d194463e2ce9e9921b9461bbd587e8e1ffce53c8caaf7e09964aa |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-31a3fc34fcdbd0b3df20e146ad8193642a47e2f07a0824b4c9adf71c08a17028 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-51624032 | sha256-4aad394ea7357f2a0f7791b24fbd189b7797fb298747b0612058f47c4bd59879 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-51624032 | sha256-6891bbe0bbb583374951214b52fca380d77e4576a64ba7c0e877068c2d33741d |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-35105ae5 | sha256-6dc84807bbc348f12c50ac33ea3a7bf088c990fdd97b595c08ca70b41d6ead7f |
