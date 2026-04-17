# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e1899273f160788957979a298d976827dfb7d2c8980b1c161a6e0c69b405f12f`
- fixture hash: `sha256-e3a4578dceff89673c40bbf12c9b294dd97be3ba2d82b9f266209970182a5648`
- score hash: `sha256-b6bd4b22e3e46c0d7fae7bc9248005ee1eab094bc31ab0fff75417164c770c32`
- bundle hash: `sha256-0d9c633ede480be06d879228717269c8eb6b0d3a0d1aa8238f3310b1d678fa0b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ecf5a06a6508fbef20c40ee36944ffad441534c7ec83a389bf5c81a0f73bcb66 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d650b12934efeebec392e80917208dc079fcbd5611721905297c4a0eac8f663a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0956396e2294b5febdbfcf41024de523cfa9a4f8d30a2fae27c3ed2f1beb278c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d4d183d1e61f67ce17d465f87a6df46aba8ea216de8eb77ae7144ba02e19d3e6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ed68c3ab | sha256-df4aead15c5f9c4e4a9e2f76c564449caf42b148a6791a299eef2e5107277fad |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ed68c3ab | sha256-3340213401d25363ae6117d7a4f6334363774d0ac5cc84dee851ecc86bccd242 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ed4a5b8e | sha256-7d19f587103c5718ff91dfc3c44f437729003dbfd86f8055a08618391a920a8a |
