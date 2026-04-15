# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f80bd1bbdcfa166ddf0a470afc83601b492c4563388d58e45f1a2c05fc2e95de`
- fixture hash: `sha256-6351998b89e93fb758f480838cac801b256b59b498a256a0fe6d16fd14c2a7f1`
- score hash: `sha256-b3dcf0a2f8d550f556ff191a1f796b9fca7fa11b37f3adec3ec4c7100c150651`
- bundle hash: `sha256-bddbbb6f13f110934fced3e0acf5a60d5b85626c4ef405acf1b5148dde85947c`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-72327e5e78912708abfdec6632c03f2d06781d0f51707306f21ba8e8eb4a65c1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-70e8087a368cf74163798ff2262361e0e82e10b1ad929868f23cc22c39c44230 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7d368cbc9a6254faf57b0e42c86fdc17fefe84b9ac12cd29c8084c41785adc22 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c7b4818e | sha256-63a17b251714116ba97a8aa906fc6f3717d05b3981824e83f7ed5a7c5944b991 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c7b4818e | sha256-de9db4cab159fb2935c4efbe194c09c7aa91bda09a0c69ad76c13fc6a6c2a4ac |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c7b4818e | sha256-63a17b251714116ba97a8aa906fc6f3717d05b3981824e83f7ed5a7c5944b991 |
