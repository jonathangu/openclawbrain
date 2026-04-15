# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c817aa28a6ea88ab750b90d075966003c4144ca68cee4de31510afc8940af725`
- fixture hash: `sha256-12c8924300be23df2d629cf06b8bf4e9466d47a9b90ef4b0770c780fb827282c`
- score hash: `sha256-0e0da30e36c44717e61639423970f86e98ad4469f47770662edc429026707686`
- bundle hash: `sha256-94f89f9602f6663965104dbdbacbcf9763f8d5f610f090d22d338a28bd3c20b6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a755e36be001d38e08b764d65e8f6dd1b01494428975ffb22d7f3f721a73e79b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b7c977ff915348a105dc87850a9aa4fa022621486792f4e2bee27264c6d890c4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8247e3832dc3cd21995bfcba0e16e9199523075d5f8e0cca53014f7e619e3ac3 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d52fb69d719adec99fe73ba6c42407cf268943ec9f4c511581ee146e824d75fb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-731398f0 | sha256-d130adfb676f09c1a8eda7fc6ca65478d7c04e0e0b80e44ed7579b220ec669ab |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-731398f0 | sha256-1f09fc9a022485f585a1b7f6ff3bec7f37cb17c975c935ffc24941f617a55098 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-731398f0 | sha256-d130adfb676f09c1a8eda7fc6ca65478d7c04e0e0b80e44ed7579b220ec669ab |
