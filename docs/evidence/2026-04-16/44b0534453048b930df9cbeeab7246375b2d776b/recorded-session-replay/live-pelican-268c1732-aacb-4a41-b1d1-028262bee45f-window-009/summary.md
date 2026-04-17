# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c4b362666455e64ccec1b7026696e6d7ee86b07af9d91203554a5f880643a7f`
- fixture hash: `sha256-ed322207ac696cb8afb94d5f75c53ba4423b96ed55d3a35abcef96ba37d6147e`
- score hash: `sha256-cba69aaea80f28abbe9d16e2837c30db38977550ea28d2d444baff0681bc3c5d`
- bundle hash: `sha256-3d7058ab8f8fd5098a1153419ad5ad6314881bfa7c5aa7f78a0862f1acb08dbb`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-659f0a773cf1e71603ffb20a5a28aebea6d5db6139dfdf0255a605e7868cd22c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-858c6a78f906fe2849ae681200f6c5df265def26c2d247e3f8b8802971e5a606 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-471c8ebf14043421240d1c6520240bf17b7c3d892ce851fe99480548cd1e07ee |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-004661552e596a4ca20ff290b2ba0f685bccba7b5edfef5b2ded04d994ca6bb6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5052f57b | sha256-e1f6bc1e052f5ca9ddb2c06fe7a93d3a1480be163e46109e350bcf7afe16c040 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5052f57b | sha256-e1f6bc1e052f5ca9ddb2c06fe7a93d3a1480be163e46109e350bcf7afe16c040 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-1ea0be32 | sha256-0729873ee0e5fc076b956259620b7d95a6a55a9276b3e3f2c8fb96277ad198a6 |
