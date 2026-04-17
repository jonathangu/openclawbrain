# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c110d52fa9d814d5415fbbb31a6466d9c241d27a71e256584d4f8da38b7870b3`
- fixture hash: `sha256-7aea2d0a1eb139bffa0a7ec4a62af3e2b3a4882d28c7223058abf7f69edb1954`
- score hash: `sha256-61c5ed4623510b7586e711f5db0da2c26fa8050def74b9e0e8bd003b58ba5773`
- bundle hash: `sha256-7844248e8ed4f3a6cb2e1e2c027d720732d6e36a4a7aad5ae5da766716f45b4f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3c530b06d54c1f577e737bdbbc2ed643a0051ad5929f4a0256034473b3d96cb |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b08d59bfab844988c15b6f5d64e50c36f535c7e6b6609a25bb31f63b53281d5c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7df6fa49c9d9e5d970b8ee75368b49371a479062898883f5d963cab9ccf3896e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d465e20f4c9ef6be0a00f20470e0a6cd8cdc9d09172dd0945c39693d39286fb3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c301afe3 | sha256-45bb99574cae944d105144165ab8a09a2134ffda24d8cf76fe7c94f9ed5e3903 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c301afe3 | sha256-cb980966a8f2be23adf3be64433e21da5d84db4a4318bed1e2e74a48ee6102c2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ed594f5c | sha256-e69bc4959159b8e691f089b4fe73d735736825a8e16be782dbc8db725f174dc6 |
