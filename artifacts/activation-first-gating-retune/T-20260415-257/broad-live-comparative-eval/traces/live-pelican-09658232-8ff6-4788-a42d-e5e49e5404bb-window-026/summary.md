# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b6b834033c55b0c1fa4b1b699a1d8df0de8cfc6135b0ef6c168d6806431da077`
- fixture hash: `sha256-da3f651615d0891ce7f18953e9f938d1c01935c73ef17cad6fec24cf102a80c5`
- score hash: `sha256-5feddde7ab2d7b24ac40aca1423fedae6785e7cd4f32f7c73d3d1172ea0ca829`
- bundle hash: `sha256-613b1549f0e22cc0aa3529d8f9d8a411a9007ec78acb7c28ff4699774ffaa47c`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8c8c13e980248fcacabcda9e687b928112b5111d3e6b7b89ab1747339a57891f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b25506bfae7c268fc779033bffe4c215db91d92b5fcdf261dc3f082bc51ece10 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f0a97934d69b510be31a6cc3f9d3951a265925bb93dc8f2b2e024131e3ce81ab |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7bd99908 | sha256-5cd3737a07f9ce28a0984a2bc0bffae4e74e9b32a1a8c478743137e3fdc47f2b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7bd99908 | sha256-5cd3737a07f9ce28a0984a2bc0bffae4e74e9b32a1a8c478743137e3fdc47f2b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-607f8fab | sha256-67180cf768cab7c45f8a56073a8ef811ea7c29080b5a10b19de133ed5eef355c |
