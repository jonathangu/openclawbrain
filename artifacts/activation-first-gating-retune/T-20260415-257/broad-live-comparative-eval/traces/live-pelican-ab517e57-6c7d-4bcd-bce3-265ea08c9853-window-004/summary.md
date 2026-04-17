# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e0ad9ec13f7d5b82b36a685375b9d3d24391406d595ab3f8c2b0e0a5247f79c9`
- fixture hash: `sha256-0d1840771e0444519c0d4b5e3c3b57cee2fa58fe3cd78cd2a661af1ba4273a98`
- score hash: `sha256-d3fd9500efa44f9c6ccac8f822c820c1be66e1fc5e58b54e2b24d4b6203422e8`
- bundle hash: `sha256-24862da47ac6e66f87b08402a3ee8f6aad0ed76dc8e99bb6f6748323b8d888ca`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-24e5432aabba4b367a9ab9972174d2db006f79b43849cb63eacaea39404c4061 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c4f92432f183bf830bbdd98b5ef8c3189d11d4bed52fa29da46e36655ecc6e05 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e6fd423ab23b467358edcfe87e0b72e351e869768e818972e3d493653513a592 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-f05de0dd5d3611f8f235605745e101ff02d693c817c93a0ded421e47fd0aa53f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0fa0f320 | sha256-c97a20d553d3c4b17dad51d255cfcf9757c7b618cd3eebe13a23b92c1a844d10 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0fa0f320 | sha256-3dbb053fe41eda4c1553d7b728171639109d65e556b7f6a844188259ec49e1ee |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-ca681841 | sha256-feb4d5b006aafe945c2b91c5fe41da0a421cbbdcf56af4e111bd9cdd584b2086 |
