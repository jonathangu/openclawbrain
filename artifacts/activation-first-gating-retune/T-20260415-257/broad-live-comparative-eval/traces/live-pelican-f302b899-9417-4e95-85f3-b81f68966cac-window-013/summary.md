# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-56c5cb0bbf3fd4c3b31b5c0ab401ad3e4676c774ca7f6d545e285ace8c5c1fdb`
- fixture hash: `sha256-77236387d32f039002239433f6a8c01de43cc1e1b10880d323ebd379dc420a0f`
- score hash: `sha256-675270d709e30e7263f846f03b370063fb7c63fdc8023463e38d950e6e321330`
- bundle hash: `sha256-9021bef4a3048ac6f82722059313a3925ba695ba1636c5b6f13aedec6dc41ebe`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-90e2065aed2d37da154d950992b20c0cd157d6f2b8bbe80fdc84eaa1f9df68e5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38c5f8e9a0e89fc655caf218da5c412d35457cbfa8f7bd7a4d89e7a75f6abd33 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-845691690424a50995279703f61066a2524ce9bae05d481543092e148b82000b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4915f85c | sha256-58c3f948d9f599b9b60bf5013ac4cb079065dc4b4c37b37c57dc09cee49d066b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4915f85c | sha256-9cc29ca04ac4909679598abbe970cc5d7940735506c9ab0ea6db6f3125601547 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fd467f73 | sha256-b73a21adb9e849b7941a5d7e0597d956a7bc774b476b9e475e3e1197970f3724 |
