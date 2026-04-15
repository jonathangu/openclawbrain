# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175`
- winner mode: `graph_prior_only`
- trace hash: `sha256-653e1762b7192e93df1dc01ad3fa2126f6513bad2e3d5a89891f193ded446910`
- fixture hash: `sha256-4262eb1c667bd83d27b33dceb3d4d1a1c6a1b57d1ba763770502ff6e7c8a4239`
- score hash: `sha256-9a83e62b5748e4c9349fb6c613022a47dc9aad2fc6212c851747a54df3b6d64d`
- bundle hash: `sha256-7fb4c49e7e26dc283c16c446e2ffe0bac6a8262a22d64b6ed1c8d963893777e4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f90cb77dd3ff507938d3ef155b0e74e6914215ea5bf7fbd610cc02d8404add3 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-db8688ef49aa0e276be838ebb7ddb7231269c62d7f73e73b2fa9d7c3bc476184 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4628d4ae6e3b9ee59df0d40749a326111f126a9626bc0fb76a8046cc9e5b927a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-93c0e518c7398dab0271172543f540707726e7bf03d6190c48c88f39f96d390c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ddd0e2f5 | sha256-5e96d778a4598e3d67e992af9d74fd5156eb1d762c2f5dbb87c3ec1610f4b246 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ddd0e2f5 | sha256-b68812cbb085818cd8517472778499f06434fd5ed8d07284a3f0799930a11456 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ddd0e2f5 | sha256-5e96d778a4598e3d67e992af9d74fd5156eb1d762c2f5dbb87c3ec1610f4b246 |
