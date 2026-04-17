# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7c9bbe1bc32703bdc0ba57cd7c2e5ba0147d232db874dca18f8d1c93a644936d`
- fixture hash: `sha256-1632c273e7fcb25c5de9fdb5adf5c07fcc4c43677737f0e63cd97217f3d6d9e5`
- score hash: `sha256-205fdb0d0783ef6287c6009db091d4385e0f4ab19c1acb47ac834d365ebf3c3a`
- bundle hash: `sha256-c5269b149fe7715abc20783133857d190ff0978b976ff7028e1417b35d86755e`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-80e9af933d3a18c2836442131236b812d1fdf8db3bb96c2fc77c951fce5a2ed4 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b6b498aa0ae062900990af284ecd30c209e4786c5fa10155ba4634af2a6c9035 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-627c378b65813f04295ae19294b3b7eebba87ef4a8499156daa51f1f5931a963 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-9c07b1fb423c855d905ab811c8bcfde76bf07d4b1bc126ecd9355d151da4fc31 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-02e721fc | sha256-1f494ca1afd94010f696c1c9cdb7f9d76c6ace6bcad72a21a27bcacd4bb36b79 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-02e721fc | sha256-aaa66ffb8b514b8e5de966f2857884d2d004095619c647be8079399ba345e2d8 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-b8988f4b | sha256-512f7506776504c8a7bf1ed1d8d9f613f16b814aba440a5bf814afdb9c3bb707 |
