# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c32e2ebb8b7b9b5fb8d50fc85f34ad187e87c05bcad5baa201a1c538d7a405ab`
- fixture hash: `sha256-e23c11c000ae3f195bff5e2ea98696c33b399e18fdc28541e3c50f4b667d3e58`
- score hash: `sha256-f265821eb99d27ac6aa55f08b1478b258d4a38a1cd997d1805bacd133e9aa7d7`
- bundle hash: `sha256-f9f99dcca4e5761269dfe0d37d314d501cc6653bcaec16053b17740da1715e27`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3b40ea9a22b9a22398f998c23b04d742ab7923c5900fe44bcea6dd68bb464780 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b1e2a23dabb8b2121e7a6d45a2c0e40fc018ee2a626be7216aca1fe7163648e0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3eb718ea20fd118d3ca2f2b461466f46af8a799b3488815b38ba64753925f33f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-08195fb6e04dbc75fa4ebf69a5d5dfd2213ba2672cf135fc3ea158019ac6939f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5c290100 | sha256-755f6f2cff46af78df003fa1010b9e72e4348208e21ad35d5a7319b3351c104d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5c290100 | sha256-3605cc6769051affbd8c95894636f8f1fdb6b6d089714b4a5958041a6b2e940d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2b4aaabb | sha256-13c88d0314a86f74acca57cb5951b635e6fc9569914b2949b72a52e37f386a64 |
