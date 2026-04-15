# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b6b834033c55b0c1fa4b1b699a1d8df0de8cfc6135b0ef6c168d6806431da077`
- fixture hash: `sha256-da3f651615d0891ce7f18953e9f938d1c01935c73ef17cad6fec24cf102a80c5`
- score hash: `sha256-6854374441f276f4aff4663ae62536583cc16d4bdb66875f86f9de72ff8dcd00`
- bundle hash: `sha256-d9bdd81faea33c0187b9b57d9429ee7c42060fa2a1bf2d04ec19f254c6e61a9d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7bb32e7c3cacde4e347cf09cbdabeddff358a8d0a73f0b6ff3c688033dc4ad3f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-477d5fa647e9456b82935bb9c61fc57458b6c7c3a757748fc669320088869be7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-03fe7e4b674ddaadc8883af0e0f1f9f9cf75560ce80b668cc4d11365912d128f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-a0036137aef2d1e335c848e7613d04a704d7ad9b78ae435bb2101cefdc81c325 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc4b97b2 | sha256-99d59a2a9e3e18fc72e66b5bb217d0b670ce600c64ac0e56e0c9b8d2c96c9676 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc4b97b2 | sha256-cd6c50d061100f23a44ad76d45a7d5a4d3188f4d3959e287792aa44f8caf4ec4 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-bc4b97b2 | sha256-99d59a2a9e3e18fc72e66b5bb217d0b670ce600c64ac0e56e0c9b8d2c96c9676 |
