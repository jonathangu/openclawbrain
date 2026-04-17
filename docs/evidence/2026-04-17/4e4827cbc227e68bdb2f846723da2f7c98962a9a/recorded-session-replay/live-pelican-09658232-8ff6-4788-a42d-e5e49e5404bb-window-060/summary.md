# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92040fed208ea65585f475c24b64fc03720a2a86a8c84eeb65240f8ebda78b47`
- fixture hash: `sha256-e789097b492b370a3cb207f40a7a3a195c61c7549ef2c7d39a0e569e0dd15633`
- score hash: `sha256-e869c48e4afc6778f95b265e8ff85d6ed9bd9a3f1cee8fb22fe09d619b50de32`
- bundle hash: `sha256-5ecaf6c71ea6ff0ce8e52098b423bda7fbe44cb43100f0981d93d0fef8c50f51`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-648eaa1db4d7048b0d51fcc33cb635f35068c6efd52393b35ef8355c224bb749 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cb6ae9aea05b21acb98d58e5fbeb97cb8f4dcd35a983996c41f2627e2a6d2c19 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e1e40b88f429c1a87ca07c0d465c90d8f4cfd260eab968dde34b96be51f94c8b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-9d0bdc1b78146238b0f993dbbe31c702779a765d2b2147a5c44cea16ccf7a15e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4a49e1ab | sha256-acf7d0355c00b1e6e2549975dbbfa74f76082d769c3163ae7a314e88b3437f9b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4a49e1ab | sha256-9a2a2ef6e66295edefd4fec3668bbb37a2c74c019c41e0718129f40af37ca772 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-648588b0 | sha256-fa54767c993526f995e4ab8569f99b34bef82f77c828b9576a692c5bf58492de |
