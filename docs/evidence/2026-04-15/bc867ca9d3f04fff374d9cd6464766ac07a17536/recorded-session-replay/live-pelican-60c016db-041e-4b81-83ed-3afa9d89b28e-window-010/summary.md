# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-838b9295d0df32bf17309a7744670eaab3129f24a6dca2ca9110c4b4940f8ca0`
- fixture hash: `sha256-56f7d90cfb38f59327532bc9b6beae4801650c72b03cf0a3e492173ea24b06f6`
- score hash: `sha256-c4d71cf0ee1372c258be1da5d26fce52a83649e3691bbe5aad865654de99bc71`
- bundle hash: `sha256-ecd37e22477a06e1c7a16c5ec78569f6e9dab47a87c6b8d5d30e50a1f0171329`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d12703af710851e5a23d60b1d20c78b1a6044ead7e09a16f607df5e76e23db43 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-95381294bba28c8c7493156df66bbd1fc7f2697d1e12e3954930476bb3b36834 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca3597c02de7488ab51188efcc4344c242537a41541f5b07e0b0aca26eaaa048 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-2e6bf6a8529ec57210e46dc140e66b1e4d47718b8382d43b811d6c8eefc1b1b1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-057ec6f6 | sha256-b4908d88cbab93984116307dd1cd4b743ad7cf4e9e93c611cbc93e2305dcdd50 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-057ec6f6 | sha256-c1cacb52f66bb0e9ce0ae9d40f5a7bf8da57c12f270468b895a4190b760d9631 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-057ec6f6 | sha256-b4908d88cbab93984116307dd1cd4b743ad7cf4e9e93c611cbc93e2305dcdd50 |
