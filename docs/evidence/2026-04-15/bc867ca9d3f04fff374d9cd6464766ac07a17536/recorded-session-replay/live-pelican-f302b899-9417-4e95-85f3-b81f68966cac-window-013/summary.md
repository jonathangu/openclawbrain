# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-56c5cb0bbf3fd4c3b31b5c0ab401ad3e4676c774ca7f6d545e285ace8c5c1fdb`
- fixture hash: `sha256-77236387d32f039002239433f6a8c01de43cc1e1b10880d323ebd379dc420a0f`
- score hash: `sha256-eb068aa35a3ecc42f967a794229a9752d9fe9af7a8440a0fac1d322a8ba08f52`
- bundle hash: `sha256-baa5ce54c5210fcc1304c4a09fca1ffcb0d06d5ac8242fc3481aa8d458f09219`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fbd60e0c202956f0282efb134299dc019cd3f8109af0e1480ffabb87e7f9ea0e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f4b16ae5207c8f5cc98c57bef324114aa9dce0f705fb72836216b69c2c80f976 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-25de4e9aa1e127708e59b78369faf66aecbb192e34330b870ca568cf4a745e7f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-23d9d21b | sha256-26c9e60d575bbdae9ac567f1ab07390f095de49038a73be821a42a2bb5d59e70 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-23d9d21b | sha256-55fae9b7fe678bbccfd38c193e029d86576130993e6afa1b46d99f3e13da4c41 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-23d9d21b | sha256-26c9e60d575bbdae9ac567f1ab07390f095de49038a73be821a42a2bb5d59e70 |
