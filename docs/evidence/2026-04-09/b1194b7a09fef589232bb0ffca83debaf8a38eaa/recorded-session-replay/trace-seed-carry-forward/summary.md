# Recorded Session Replay Proof Bundle

- trace id: `trace-seed-carry-forward`
- winner mode: `graph_prior_only`
- trace hash: `sha256-21ed33015f7a51ad5fc95030cd8188dbc6655b191e53b59bec58141493db1904`
- fixture hash: `sha256-912c47fa6e90f710951f9473e5540907d6ccf58746703d4574c6d5fb9a0dd66b`
- score hash: `sha256-03f3832f4899fab14dee8fc6516a747f11ec6b5a3edaa3d2985a7350fa6e8713`
- bundle hash: `sha256-af81643c94d1522cc456e82d3ff6575cfb4cc3fefe0017c3c35876eccf79b197`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 6/8
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/2 | 0 | 0 | 2 | 1 | 0 | sha256-d38ca63ee2f4351248154268de4a796b6eacae65de64f0d136e785ddc845ea9b |
| vector_only | 2 | 2 | 2/2 | 0 | 0 | 2 | 1 | 0 | sha256-639d9bea45c6530518736e6091a81f5cc8c9614b7fd44999639eef47174b5474 |
| graph_prior_only | 2 | 2 | 2/2 | 0 | 0 | 2 | 1 | 0 | sha256-7cb672d1f9376847ab597da113681ceb351c7e52221fe6a382e861edaeed998b |
| learned_route | 2 | 2 | 2/2 | 1 | 1 | 2 | 1 | 0 | sha256-6e99722238fb628b125dda525412adedce48428eb10ce2583529907f06d708fa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-2 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-2908fe24 | sha256-4e40c09799ab1622738556192885550282d7434150b449fff247594b978f6fd2 |
| vector_only | turn-2 | 100 | yes | 1/1 | no | no | pack-2908fe24 | sha256-7dd715beb5c9718c8700bd9708b51e12c2956973d2e26c840610ec00e051155f |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-2908fe24 | sha256-4e40c09799ab1622738556192885550282d7434150b449fff247594b978f6fd2 |
| graph_prior_only | turn-2 | 100 | yes | 1/1 | no | no | pack-2908fe24 | sha256-7dd715beb5c9718c8700bd9708b51e12c2956973d2e26c840610ec00e051155f |
| learned_route | turn-1 | 100 | yes | 1/1 | no | yes | pack-2908fe24 | sha256-4e40c09799ab1622738556192885550282d7434150b449fff247594b978f6fd2 |
| learned_route | turn-2 | 100 | yes | 1/1 | yes | no | pack-73c9182a | sha256-647ced092b35d0c83c87567537668c8a74cda281fc9808502b47a66a0e6ebeed |
