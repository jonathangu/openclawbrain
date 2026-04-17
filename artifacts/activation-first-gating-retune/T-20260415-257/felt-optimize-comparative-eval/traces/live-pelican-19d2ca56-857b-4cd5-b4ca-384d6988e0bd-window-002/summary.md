# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3587f37a965ca48e3b14fe490f41619f4a64d9248201fd791da49328673f2fd`
- fixture hash: `sha256-72b30c69deb757e882e827610fa9efbae23b0f6f41cd081abb4ab731c8f4dc73`
- score hash: `sha256-ec724b3ad054660583840b34b688d800962d5e2cd92acbd1100053f4e0820cc4`
- bundle hash: `sha256-8aa422c29f02cb52965192e5fb96a1d63d0627d0f5f783013afd5d73f552ea8e`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c24e3fd46da0eb608623bf9cdb0e333f48f37ee134ad82a8288bb803d8eaeae5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-56ccd6a1fc0d32442cfe386f42959f8c5db1cc1ad273745f29207601b2a5ee2a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-da7c6140acefded0312fdb3cb88e22c57f7c5404e2b6813876bce105dc131f6a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3c645356 | sha256-18f96eb05374f9d2ab0d1131b7d4421bc4930b70a2685271355a912141afe472 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3c645356 | sha256-aa9f99d3a5c42de0904305aa260b742d0d30c4864703248322c59137a8f6d6b4 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-438fa83b | sha256-83b829a5e7c955700b06265220ca81a77392f4736a3d1775cd0acd8370d35545 |
