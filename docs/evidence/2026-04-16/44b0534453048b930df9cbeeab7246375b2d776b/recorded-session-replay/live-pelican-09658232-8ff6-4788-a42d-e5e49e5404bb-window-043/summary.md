# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5dd6be200875e55287b6b027aec6adb1211d2b987b83c6a37b985516f7118529`
- fixture hash: `sha256-eec602e66445ff4dd47c7240e799fd3d8564ee87f3fa97f5e6b5673abf356c14`
- score hash: `sha256-a7bb78ab6fbb0952e08aa5d4636c70f592b033b093fcb9011e60f834a38bd652`
- bundle hash: `sha256-431bfe358d32f49611acd6fd4162fcf9a7bb51199f467e155c1c499fe9730e30`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d99ddcb432bf41d8fa10f8ab6904c40f835adbab6565ab293b9f4c7f5ab02130 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2f439b91d5f48a25c8bbcf87cb86e5f28c559c4e52ff0628a11e81c0c39c28cc |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6844a80c09eabc4e6cc51ed591c8bce576bd94b39cd9851619a1f1e50907e131 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7326660b1db00058d86cde26ceda904af89d9b49c324eb5300a9ac270ca8d749 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5b384764 | sha256-3e1bad89190458ce2ec5af33f719c961cfd8d44a7e459ffae7ff5fd79ffe5b61 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5b384764 | sha256-44e02ff29db7d9d0f1cd7d3d32da0c333046f9bd6bc67c9e350f5285a6bef390 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-605eb885 | sha256-a264af4aea5fe97c1e603fbc786dcac7c49220a88661f265518c60f308e73b90 |
