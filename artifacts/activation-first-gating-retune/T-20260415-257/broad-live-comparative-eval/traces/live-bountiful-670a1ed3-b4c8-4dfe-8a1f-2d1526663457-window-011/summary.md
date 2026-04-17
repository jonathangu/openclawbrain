# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2d785c91c6c2597c88bfdefe91898000c30733ecf3cca8e1fa5fd2d6621049e4`
- fixture hash: `sha256-63b7942b83cea800c5fc9cb957ce0307322538d9d8e1a745ea7ab80b74e65911`
- score hash: `sha256-0df531b0d534cc5f26675bba80cd6bee9de01043b4ea3a609ec5b7b3242e4786`
- bundle hash: `sha256-2cda03ea5e7df52a0e275b9b495c07efbbb29b28234b8d12631533f70b32b2ce`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4648c104e20ab98d8928f41590949536cf65a6240f7fac95811ce6126bd169f5 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-9ddd9cbe67a8267b3e7364cb223c6420d81404fd4443c7510fa6332e99b308be |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c9d55e409867eee6c3b4cb51395a8adf1a85c42c4e801c7f9813f77fb94d420 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7e8ceee82095c7665e8331edc1ca978fe21e8b3aeaec255acfbd325dc8f0e742 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-f296a7d1 | sha256-4aebbb79442e115d31580a0868436a67cf0e1e4f79f7932b585e204f0bd21846 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-f296a7d1 | sha256-f018b8867fed291afa291d619fa40c15b4639d740105d9e2c3d17eccb24805e2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-54e903ec | sha256-365cfc832298e51ab541937f4fdf24b5d4a6f3a3957bf23516c81280f06bea79 |
