# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f65f7ec4c1006917225f6f3df2297434078972719c9016d9a4a28c343601c090`
- fixture hash: `sha256-0846d04b26eef0a1a7c06190a5a1fd4f54e0a1ec3fcf3231ae0df203565132b6`
- score hash: `sha256-6b0e71cda7724b1fa7479c6f2c8d2cadea8aaefa4f06604073849429a4476017`
- bundle hash: `sha256-8551d615f6208b98baf193d33422134d1dbdcbe4466a37eb7087db3dee89fa52`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-42f48b48c6c450f0664e256db3a267d908035a318a1c9a74a979a0b9949d1634 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2e8f1a6a45ff4cd5a695a270a5c4b37c359171b90ca995d48158b18cef08b176 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ec174c8b47b0095e5df4a0890645dfcc0626a7686ce3433bb5793ad694c944cd |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-578ed768df1b7f488e9eff00791e0ba206931e847b8c5ac95b8de74df37bd3cb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4ba9ffe1 | sha256-786ea83889971cc8d2d7bee5c9cfbb1aa4f15c41c6eb04dc6eba15f536f3541b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4ba9ffe1 | sha256-24f7881a167c1378aafced71bf1c4c7d8a3fe9b30a3f888be3fe673ec75e894c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-4ba9ffe1 | sha256-786ea83889971cc8d2d7bee5c9cfbb1aa4f15c41c6eb04dc6eba15f536f3541b |
