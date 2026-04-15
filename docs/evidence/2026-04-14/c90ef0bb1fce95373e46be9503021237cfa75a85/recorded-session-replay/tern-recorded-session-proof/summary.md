# Recorded Session Replay Proof Bundle

- trace id: `tern-recorded-session-proof`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5bc5fdeaf5c970cb29b8a2da8858f5dde637257b3ef1f2d1015b4ae096a5f09a`
- fixture hash: `sha256-be9ca6cad4bdbeaafef84b4f5f64b804b10095017314e91f5b03454346dec6ec`
- score hash: `sha256-a7453179760d6730c4cb8a8f78165410e432147308627d2864232ac3848066c1`
- bundle hash: `sha256-545759a1f367f1a420be9630ef812f9fb28f751b3c6e9dbf9a38a9a38e89dc93`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 12/16
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/4 | 0 | 0 | 2 | 2 | 0 | sha256-5f50e37eaccb224c1a52f95bf6f77a322195f8a2bdef00652250ad97536854f9 |
| vector_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 2 | 0 | sha256-3ab63730cb954ccd13885a92455d665d5af5c0c6eedd0db3fb063edeb48a38c5 |
| graph_prior_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 2 | 0 | sha256-63b4c0b589ce74be372c0419c11e53876331cc7036cad0cb0dc50d01c748d8c9 |
| learned_route | 2 | 2 | 4/4 | 1 | 1 | 2 | 2 | 0 | sha256-80e8acd84d67a5d4285809395fbed470955174736d369a346484bffa50b5c8e2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-alpha | 0 | no | 0/2 | no | no | none | none |
| no_brain | turn-beta | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-alpha | 100 | yes | 2/2 | no | no | pack-5f82debe | sha256-fdcb73b2e2cb04b4a682d14b649e74b9d3b5b1fae9f7dacdbd5efb8f3994be60 |
| vector_only | turn-beta | 100 | yes | 2/2 | no | no | pack-5f82debe | sha256-34fcf01e623b7cb6bf32c3661b8407ec1ec96b46d00fdc65dffa76ec9c456b33 |
| graph_prior_only | turn-alpha | 100 | yes | 2/2 | no | no | pack-5f82debe | sha256-86b941fa183071a3de84bd7ed8bdec04b878b655530b1e844d3a31ddaffa8854 |
| graph_prior_only | turn-beta | 100 | yes | 2/2 | no | no | pack-5f82debe | sha256-89c00720d88a5cd04bd011c134f889d057f3a76b2da7600fc3d7629e41da3414 |
| learned_route | turn-alpha | 100 | yes | 2/2 | no | yes | pack-5f82debe | sha256-fdcb73b2e2cb04b4a682d14b649e74b9d3b5b1fae9f7dacdbd5efb8f3994be60 |
| learned_route | turn-beta | 100 | yes | 2/2 | yes | no | pack-ba1ef19a | sha256-fd5b88c3b35153143d3eb703210c4312e3dac22854e046955565f40bfd931f94 |
