# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0a0997478c28e207b3d7bd7d15ba48de1f89922b141d32d3371eb242cceef4ae`
- fixture hash: `sha256-9703f088aed39f3dc293adab170d3ce2900e0f982693a357e7c4d414d8997e11`
- score hash: `sha256-46eb6a3baaf612c0f0ab1b28106383a82134dc6d2215c62598e373e04c95992b`
- bundle hash: `sha256-d22c04d300ff946b5decb3afa3520745db0d2ea2fce684118189f63f522007e1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 1 | 1 |
| graph_prior_only | 1 | 1 | 1 | 1 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-27bd47690a786c5153f1de6a47c4efb1e5b3279455c8d667f6627f41a8eb28f0 |
| vector_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-7a2a5bc1d4704848a6da775c8cf84617bf0c81063bf011848c9bee2d01203051 |
| graph_prior_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-aa21228d9cf725ff38e594c06a824bfd12bb0b411ff9cbc2bd50e11291fd8cac |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-9dedeb53efa26e02f60826c2c0ebbe12c768d3f6fe506c85a38d60041a97b44c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-2d8a0be9 | sha256-05219344423a0068cf9c5b47c8c3fb006bd64f83369f8b2092dbab11b3d4ceb3 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-2d8a0be9 | sha256-fa8bfde8c758188e431151f59c8615f40b77078961059348aa84647c252c21cc |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-2d8a0be9 | sha256-05219344423a0068cf9c5b47c8c3fb006bd64f83369f8b2092dbab11b3d4ceb3 |
