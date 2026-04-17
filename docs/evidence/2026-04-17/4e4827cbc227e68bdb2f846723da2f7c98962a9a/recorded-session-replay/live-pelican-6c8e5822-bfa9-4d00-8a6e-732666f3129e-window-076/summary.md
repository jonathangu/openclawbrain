# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6f8a152004f1762e8eed1ecc9dffc7029171fd911a8be7a9ecd27602349fd8ea`
- fixture hash: `sha256-fef2416147c50461e059554c89ffc13514d9838c25717ac0bb496af10eda074b`
- score hash: `sha256-629bca159f37c4ad804142c7bd186f0a540543b9b3423a74daf3f64fdfd49b24`
- bundle hash: `sha256-ba2293b07cdddaad8d7243e58a206b25917233a2c0b21588cedf8e941a36b03d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-510726e0c2b1103191bca21eb76122edbdd44953bed132c7c6febf953ec52703 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4cd4df94c7ee038b6ffbb18f1b2ae69c3470c7fe229e6a6167f7628da94d0dd6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cc2cd1b45f64ef88096f5fcf3512a3a2ec292f0076aaab21434153bbc45c1bd0 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-3f2dd641fe2980ff57b7703eea41cc15fb812d8f2ab68359b22945d1a02b9d8f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3022ef90 | sha256-4b75a7da8593770dc919d97b5f8317262d65d09e80adb62fdb52bb653386c7f3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3022ef90 | sha256-a7c47372288b9785e491c4097ed704e051880a2738e087c004899259d671e970 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cfd3e0ad | sha256-1c190b4edebcc2e13718fab43582416ae3ce34139be140e2e8a07ba8349ed068 |
