# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37ca78c4f79af1f5ccebd457f09d9bd9f0b270ac0d1c7dc3ef10aba20d199a04`
- fixture hash: `sha256-221b36f5e3c3b83dde39237b8133ec3e68acdd74bce0b4e3672a3fac84a8cce9`
- score hash: `sha256-ba16caa1231b60f24b2dd2ac63a1596a88b097b29445bf59dad224cda6811c19`
- bundle hash: `sha256-38da7f48784b5f60e8f3b9f0489f39d552d307153381129dc27e962b7f8d6a03`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fa6a30b0b756b7163e1ab0f1526218df1fd81b134bd908830d7627bb5155f717 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e1fefb672d180123341542bfc33a8290d3d3876972bff370b6b0540a612fdca1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-609cedbfae7c38c90fd3ec209437891aea902679ff9c34fdbfe1d1a4adbdeb62 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-b3868ac2d6d45d2d032c3306967d9349ac821046e0f85caa4de4d96cfe4bc1f4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7890667d | sha256-b29dbd3072aab9799a676dc57069304299e887f38978cbb1b6e1326774afed10 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7890667d | sha256-c7743a53d61c5ae63a573841f0a205a16f99941d3cd0f5a73b13234f78c6b154 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-6cccfec0 | sha256-a2c6564e70558ddde6c970325f5598026349f03e98267722d6aa8707a48f70fc |
