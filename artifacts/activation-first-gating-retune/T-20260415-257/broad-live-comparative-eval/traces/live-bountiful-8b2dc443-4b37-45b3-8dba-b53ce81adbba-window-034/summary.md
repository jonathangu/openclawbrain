# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f80bd1bbdcfa166ddf0a470afc83601b492c4563388d58e45f1a2c05fc2e95de`
- fixture hash: `sha256-6351998b89e93fb758f480838cac801b256b59b498a256a0fe6d16fd14c2a7f1`
- score hash: `sha256-4ff1f9679bfa7e91dc41e38da786ba788b354bfb73aa8349f781c4b8a5d8fd83`
- bundle hash: `sha256-8e4e1366395c88f21105cd6510e2edb5c422f7dbdbb267b664eb1141eb5ec8e6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bc854933a918fb32be2808e49cd62ec70012cfa090e09597f270649ed2f5446a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2051ae49a6699ae5cfa96c106c75a0dadcce2ad3cd2374896c6465204879b32a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0a4d5c287db2ccc927034aa5822caf388ed3f8532c20e370a2c4efc5b829d08a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3a5bbae34c42534b4f1e94454e4265e65473af53a73a894c700a62a9519e65a9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-13cfc819 | sha256-ada2642e8240704001262302acaaa8aa35caede4bed80af6c0d55755cd4ea2c3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-13cfc819 | sha256-068c22fbd353798f703dc66a33e07c6d1e160b956040697e6657e16e9262b5b0 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a0988226 | sha256-3a375c0af660045d3c6b84df40739e87277f6f4d6e3c0ba0959460a64c481df7 |
