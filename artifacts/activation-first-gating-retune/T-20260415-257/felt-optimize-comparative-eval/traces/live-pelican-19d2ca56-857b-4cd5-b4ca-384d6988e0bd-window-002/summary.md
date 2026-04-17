# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3587f37a965ca48e3b14fe490f41619f4a64d9248201fd791da49328673f2fd`
- fixture hash: `sha256-72b30c69deb757e882e827610fa9efbae23b0f6f41cd081abb4ab731c8f4dc73`
- score hash: `sha256-bbcead4f9194237e74a7e4dfe79421aadf213ef4ae8338935d1fca6f8d298307`
- bundle hash: `sha256-0c4e8e0ed30e1ee12d7c3bb0a5477fb6a670aa4e0914e52c17c6850c57f0ae50`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9cbe4dccbf152c20742a7a0d9f6d7f345aa7c7916722159d8bcf4f7a084bf5a1 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b35847a3a7ef4d976636b441c31a892bcd94b15b5091dfef2527188367fa9986 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f302234bb074b9db9b1838460b1fec18ffdde0b85470a8f6788570eed958a62d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-fd47bb28e7f54739462d913399541d0f6dbdc5578cc1bae0cfc1e982884c1f8c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-521a59b4 | sha256-e2cb9025fd362236e8df234cf5cb08fdb973d03c06c9ecc8a8baa24032463c5f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-521a59b4 | sha256-47a88aa79fc7af02fea2f028016bfd5102d4a7544555131bace9ceed0922056b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5945ae99 | sha256-c436f5b8006b7cec204290f75fb085744bf9439441d47b079f867a1d8e623349 |
