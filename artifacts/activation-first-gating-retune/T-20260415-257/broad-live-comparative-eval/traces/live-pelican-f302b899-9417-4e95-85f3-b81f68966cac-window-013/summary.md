# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-56c5cb0bbf3fd4c3b31b5c0ab401ad3e4676c774ca7f6d545e285ace8c5c1fdb`
- fixture hash: `sha256-77236387d32f039002239433f6a8c01de43cc1e1b10880d323ebd379dc420a0f`
- score hash: `sha256-f54a121234cb9728148c6c3c76af8d54adfc5eb30cd75a57d50d88380fcc7d5b`
- bundle hash: `sha256-3966ad9eb2dbf0cc8b64de278b113888bf032ca7e143b93dd6787bcf65ca4a36`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4dec047b876b4ef1cbff2ba1d3926376bc0c710b4b08c16a2a7795d5ae337d56 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a68cec14a4230afa48ee18c39033596804be6157417d3ab97cdedb1187595c5c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f433dad8827f816bb937a56f07f3272f289ee0e4efd7f7e13a14ba2676bda72f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b28fa0bc333e186a9000c6dc764a78d7afa517d898dcca3c7968bf795381df44 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9b2a17a5 | sha256-3bb2c85b0c5013421e050cbc6b0d1d795c0ad5853a3ece2a4759b8d735bc05e1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9b2a17a5 | sha256-a9386becc6e5e3aabe34b44c24da14988be9424703d3e5d2cd92efa30bb68b5d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4f5a9ebc | sha256-372526d6720d8b033d21573f72e015d8da77d010bc347ffb76c724bc5eb9b8e0 |
