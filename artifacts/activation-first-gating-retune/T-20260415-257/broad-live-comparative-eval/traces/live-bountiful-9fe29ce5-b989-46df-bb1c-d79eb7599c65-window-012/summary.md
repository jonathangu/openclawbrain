# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8cad3f46eed815106d3bcaa6e251d3bdcbd2748a5a2e05af3c4a013db9a57004`
- fixture hash: `sha256-0dbe5f4100592cfe93341d37b8f8c029b314e4248749094aab73a6cddcec834b`
- score hash: `sha256-7848d1008d8e7971565bf742e60604516f38f32d21cd6af555d83b1996da3388`
- bundle hash: `sha256-573136fe2689b8d081464d2f11acb6304c88d23c5c8a1455cae02e24a2058c57`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-70b89b4c1dec935305e4546f6cef47f2e70c9b2ff0e3d82f19d8e936c0d67142 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-42030635a7e4a9f1a07b562352d060b383d32cc2c82fb70fc81de8bf09ac8f35 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-090940efd96a1265b908e303da2304b04cc89760995f3f5ac7b768456fbc5fb0 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-9b9413d598d94e981a4448cedddc137d68cfa20f5921598bb08eb1af107f6cbc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-638c601b | sha256-c7e12da679dffdc5bce8b1df25770d30ed4b57b8ed32b4068071c34eaa91624a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-638c601b | sha256-7392c569fa8c364482e55c1423d5d9fc908af60931d5067c01aa4277e4f1d4b3 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-638c601b | sha256-c7e12da679dffdc5bce8b1df25770d30ed4b57b8ed32b4068071c34eaa91624a |
