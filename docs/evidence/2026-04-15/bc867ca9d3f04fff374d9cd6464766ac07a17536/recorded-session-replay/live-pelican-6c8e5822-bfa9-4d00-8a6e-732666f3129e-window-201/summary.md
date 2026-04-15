# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0daac77494acc6aca056ecf3e9f12fd58f33e4988f5a646f0ba5dd6ef080149`
- fixture hash: `sha256-267dbae4de2075656207ffdf48cdf822d6c7cd1996c42f8535ce786c53f3660f`
- score hash: `sha256-7a3f82d8657a15a94c12988ef1e1f59212bdf6c7f7e53d7636ef28567aecbd41`
- bundle hash: `sha256-34532029e9af45592c2bc789ccd590bf57a8eab402c72b2a4a309f4e1010551e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7fed91035473e3fc2f9947814563b07b47451ce7ca5cc7e497e3f1c68f58c389 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-48db65c0f44708f8f77e60eaee62bab2d81c91bdbc8f59ad7b7a7ce54a3940af |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fb86d550c57a52b9277d7bb7ec9a47a2f9d8f1a69c9d2c0763568487b5103d69 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a91643788060747712dae949b7c13ac3e70da7adde7e13e08c0f41662412e1d0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e9634a4e | sha256-cebdcc1c02122600d096ea8988685cdcd77c5e9af5f1fde22dab0f4faa7cf5ad |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e9634a4e | sha256-3f696cf2fe1ea0463eb3c8a27f293dc9c687de105dda13a2e7a50c4b0c6f51f5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e9634a4e | sha256-cebdcc1c02122600d096ea8988685cdcd77c5e9af5f1fde22dab0f4faa7cf5ad |
