# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cacf2324859afd8e6f3cd4cc1393b48174ec7965442a67bc34f8b6260b72a625`
- fixture hash: `sha256-ca2cd496b9308f9d13fcff6478fd7a04f824cb026dc43bd11af171fcc1a89539`
- score hash: `sha256-2fce89011dbd28b21a9641aa8d21326a75c114240df7cdd22ea33c62cbce78ef`
- bundle hash: `sha256-1e10a25cc593ecc04775d3f9174223457d49fd1dbb44467151a6613403ef62d3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-591cbecfe0bbc6c84d3223d049bac9d2eb96d473137d7ef277a661d0bb2ceee3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-89bf4b1a77b01662c41a6ea924240dec8bba288100e6f1bd861d0d17a9306d76 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38c7299d0b782bace7bf65f2e887b142180b50656ae38fb26ac5f35d45266e34 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8250d2c4701221950f476d8afe882bbf0ae4820c1d6dea6485082b67fb1eef8f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f4a52ae1 | sha256-af9a1ac67928208240b31b52147a25831e93401d1f00bbfab8ea4e7ec6791e00 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f4a52ae1 | sha256-508fe5c9fb1617a66a5ecff541753f01688179b00963902d81856e0a4c530602 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-496e5fe4 | sha256-3a478ecf2e4a3b448d22a3b3863d75f7da61895c8c0f05b9c42e8cbaf888aa6b |
