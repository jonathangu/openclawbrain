# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd603b557f17857772a54a908d5f0ba5df5b9405e501fb3b65a61cd496b30680`
- fixture hash: `sha256-a72ba01cb2a77634727f52aed3858de560e72d44f77e52442f91249de387c84b`
- score hash: `sha256-4bac93f278ae7ccd08eb2169f98bea7af48cdfd808dd86b4731d017a10c3aa34`
- bundle hash: `sha256-fbada315582c0fc45e53d558ebb097e2ea153e8817f952d894db8160736ec1cc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf4126f9f6708337f7f0c62a2db19e988397589b870e571ff16dc3ae73782dd0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-098b9a610ef7e74b6c5c284ff14415a099a9f9bc1e138417341215cfe52f33a6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-147b62d812a06e712be9c0ec652c1f7c51e696234bedeea3ba0cef9fec429e9f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-df5e513d41346c9f05fe766aa5440e52f131c019063804e12d363b3916df4a11 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3b2456e2 | sha256-677fcc9eb4f8574be4cd902565c79b091798e614da34a30f19a64ff504a13a09 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3b2456e2 | sha256-7c8dda97ff490cb5eeba86250e7d40cf10fec79707be8602f5edd8ce22730d3b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fb0fae15 | sha256-78bff32dae80c4bf5d7d878569945d11b6b85405112ae3f19ec16f4329353ab2 |
