# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f1d2242059d4efafadb64c6610106d98feb5c5f961bb4e1b42df971f4c261a60`
- fixture hash: `sha256-0fc9b302b2dcdfe1c12ce6973a204a93e663e9ebf3a3fa850cdd1e41f05e02a3`
- score hash: `sha256-77ac7d0bdf28bede2e0181fd77714c481d690ab4a13d81dfb396fd0bd82b119b`
- bundle hash: `sha256-f9cdede5b8071bb18e0c0c368ce284ce33f1c8e7d9a8022d09f62774239a1a81`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1459c2fccdd36ae63455329e95c444cf7e45a5cf69fb7b55a421593b88bbe48 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-902396b71f62f2cf376b80ec109b2f9103278aba5b666a489b6e8e41213caabc |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-3f2f0daed5389cfbfb62a7981ea46e498e106999cd5698212c1d574beeea8910 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-fa48b1d2b34399d6845a361615d4a58859220fab1bc84b71f9d83df9f2a7eb6b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-d46f3f79 | sha256-1caa66ecb24957e347ac3a09ce3d881a5b79628034a6a52a9cf95840f29b0348 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-d46f3f79 | sha256-8631c08832e800c0b1041edbda67fa836ff5663a92869d7a32a475d77fc638ee |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d46f3f79 | sha256-1caa66ecb24957e347ac3a09ce3d881a5b79628034a6a52a9cf95840f29b0348 |
