# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3d634932f1bce6d4693c067f779badfb747407b8de4f4dc108015f5fd2e78b`
- fixture hash: `sha256-178c9c975a3f9bee04b778ee3424e4eb908e1106cd7f867502edc61b1de425cc`
- score hash: `sha256-5d2048260f8b5f8f6012b031eb72b0413214eefbdbbc93d053867f7ababfc5f7`
- bundle hash: `sha256-ee54869159986bb5b63f500b9b5112b35b6383194d3ace2db2f127479e4f2c85`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-252786d108eae05056482d31aaad41cb1fd7abe9a8bca72a4a7a00c78ba84b59 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-39f5e04d4caa11abfae885370f800c8af720d1184f427e3f5d93758fca3f5919 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-874d5828ff5a746f166719cfc417a1569c56a8a18cc21df28ba4d9bb2ed3e97d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-7dd6cbff375956605a0050db10f827d0dc6ccfb47e1efa7b87c37b7bdfb2b17b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e302577b | sha256-95c80fe2c1b00179301f5c18384d5dad6b52b5674d1e50dded6ab46377988711 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e302577b | sha256-afc2f619d05edf6539f98e051634aacae56680695f3dca4c447973ea826d2529 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ba8f9e5a | sha256-74d0912ed77acf803d1128ae4b2962b41f0693065dec76e329bf53e68bbd5311 |
