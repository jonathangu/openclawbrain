# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60561f5cd4b9679d1d07ec70fb93c8ce09ef36cd5a40b0352b67931141e9e246`
- fixture hash: `sha256-d68e77ff5a53346b0fae859928eb6131851ab9f7d88f52a94509c0f85b109391`
- score hash: `sha256-038655649e5304890f88adf5c7c2a0f41110cba6f36dc6a4db0ce37d960148e4`
- bundle hash: `sha256-3507fec4901250263be2228878de08ee631ffc9d7aca189731e1b65915e885d1`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-350a961987884e875cb36c29ae1cb810ef961abe38158c92bab3e2c95369cbcc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-73b2da1ffab50c6eaf31495c79e6f0d9aa73f08583bd9eb1505a9da98815e26e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-45b1288d9748724b66609afcc143d8faafebb2b39f0697a706185256f7ba89d1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-7d1ac9a9a6aef3c0a99832e73014767bce47d7214010a76f052d3b366df9f540 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d280f5d8 | sha256-a13a915f770bb8920aeb31f3ae622a36a988f499f51cf25d2075f40f5c12f262 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d280f5d8 | sha256-52adbfc822add42f9e61b7c18194352d7370a42ea9a528363927c1f4352fd6f7 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2a793725 | sha256-f849a929b5ae15995f9834aa8fcdc3a5df29987bfbee8db59c93947abac8fa4f |
