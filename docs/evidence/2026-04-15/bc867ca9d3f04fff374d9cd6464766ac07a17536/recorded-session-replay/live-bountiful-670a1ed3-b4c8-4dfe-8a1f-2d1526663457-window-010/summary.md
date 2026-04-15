# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c80a3decbe06cbf3c4af187d8a5af847ce341540f23d409b6e7d63d31df4bcc4`
- fixture hash: `sha256-741cbfbe2c3d2f3a4ab8e97bf7b8405a7d1cec581f3191dded735c7802b1e00f`
- score hash: `sha256-08f5693e054643a28b345f1525b6a6c2d5899b63629e2fe75ce4f6dd40d85713`
- bundle hash: `sha256-b38642163da0153c910a5a97dac89b482a18f79f368c8fe63c12def45c0156cb`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16c0b7f4b283cf0cadf9518aed3354f26372dc3c9867fbbccefe14e243137800 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9dc2a0f39cd1fbb9afd2218f39195e3c8ecd0770a732fb6a108d345aed7a1199 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3f429fd1431b2b95989e29db1ebffe633617eb93ecce082142354c5022e62067 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d430814cd16ff573a1ef22dacc2c3d2c45e8e6a0d691a352e863b6546f3165dc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-793b1dad | sha256-9ec95911b8e3f402d013a0d1334cd89243f25fbd68cf3259af098739496e8f4f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-793b1dad | sha256-6bce0d691061380db354fb8c4b64b18ebb13dac65d7c8a9ef70cf1e465333947 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-793b1dad | sha256-9ec95911b8e3f402d013a0d1334cd89243f25fbd68cf3259af098739496e8f4f |
