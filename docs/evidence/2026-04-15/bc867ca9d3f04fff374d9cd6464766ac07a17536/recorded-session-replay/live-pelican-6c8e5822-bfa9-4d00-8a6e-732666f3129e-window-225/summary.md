# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7768c82b82cfc9e79b11d2862950b229d848ebccb180ebcb1860140cf56b1f18`
- fixture hash: `sha256-e3d4346656fea9fcd52a8093d89ccf43c79e719fec02594aace8851b57c7f190`
- score hash: `sha256-b83315b3abce8ed8553d5d68b145dcd5a177ea535d88e037b540094b399666e3`
- bundle hash: `sha256-872009e1ebade59a85486ce32fa048acae81b2604de1632012a3b73301c46daf`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb4473107ca7b170cb7198e9a132dfc26d383b8a4567d404be160b76d2d08390 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a5d09dabf7e4ed9715e95553d6c1f631086300bd8f5aebf1b7cc1cb7cd54628c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-66736571da7611e4c1da6f8ddaa7729c57f8ec2b9d31ece3b27f3206cb2a1031 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-bd431abc8066660fcac553853e416e7d853d45fc0ee6d063514867b031f9b982 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e8c60678 | sha256-c9a38c171324166ec3a507bc971df9fe6cfe7063f5f761b97d4cfebefc0639b1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e8c60678 | sha256-7d7c6f9ca356387b3c7fa608e5592a5cdb30eaedaf5c4e2b6034e73cb05f6bb7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e8c60678 | sha256-c9a38c171324166ec3a507bc971df9fe6cfe7063f5f761b97d4cfebefc0639b1 |
