# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cacf2324859afd8e6f3cd4cc1393b48174ec7965442a67bc34f8b6260b72a625`
- fixture hash: `sha256-ca2cd496b9308f9d13fcff6478fd7a04f824cb026dc43bd11af171fcc1a89539`
- score hash: `sha256-7b492971235f66623ca81eab67c63d1bbf7c116758741efab00371e724ee7689`
- bundle hash: `sha256-614315e2b9e88389664d26602308cbf08441aadc40b99991cdf15687894dcf81`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-591cbecfe0bbc6c84d3223d049bac9d2eb96d473137d7ef277a661d0bb2ceee3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94cac5299716c3ef2f4cd75c79a726022aa2d8f01983e9687ecceaa0e1ce4711 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0a429b271cbdf3e5214b31d4d68cc68c3285324f8ed2cfe210f958bc75fbcb9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2a6b357313c893f7310508c4ce29f92bd9ebf64f44e1c869d03180c394fd47df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4a5c643 | sha256-0b75d3b5f6ce1f323911bf0423c81d3fa1bfc6704ad76df8c3709a9ec8f1f5bb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4a5c643 | sha256-e3a665e017adf153df951f7c1723218c818eaed49030ca1d3abf24d1eb698596 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-296efb46 | sha256-1082a9695a76bac81c48ad48501c42537bf7cd0092dce129107eca7ecb84b1e2 |
