# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-766d9b6ce430d9d07fe2ff3297e9849fe05332d7539d3d62db1cee2a9f89081d`
- fixture hash: `sha256-21e8a90c2dad8ab78ca636bf0f382e5b550e2af76a7681917f1773769c731648`
- score hash: `sha256-ffe97872c42ac7422692e54f5dd3fb05e580bbbeed674eb3e8dace41072726f2`
- bundle hash: `sha256-54ad65ef03baed49453d57fd8783c6f6afedd44c01a3739c4238f1330bd4f6d6`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d8021a8424a98c9c0ae913d23bd911fe66b4179fa226e5ae4873cee34e53cd89 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c3597d4184778c75c292434ed09f221fe0d0d7e13fa378f99c9354c193a18dea |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-89702f02faad76698876dda3b5e77dffb3b63a3c68597ba0bbbce787a6a3cad7 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e940a17dc6b8ededd2146469265e19d53337d76122279e2efc773c4794af21e0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cd9366dd | sha256-7c4aa6e8fc86a1893600a7220540ee08a3400415e3388c6dd8d7e49a0515f8e7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cd9366dd | sha256-786895d679363f68a3980556e20cdd8c8771b173b6570e8c76836f3eeadb807e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-23cbbb50 | sha256-4a3b4a1bd6de2d853c8f3811106ff872cba532398f18c8e2f590a0fd96d0861b |
