# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d5b2e9dc9e67decfaf3c661978d40b3965c717607588db6b26b950194e4e66bc`
- fixture hash: `sha256-05d4de9ab3e3c70047bcf0e08acaa0f5e5762d96334a591c78e4a27669a8787c`
- score hash: `sha256-db420862008f6faffa8cc9a4369b30bc5840656848734918d6456d424484569e`
- bundle hash: `sha256-be1e1c6f9c81c4392c98a1d19aeea1344599e7d45d72f45d5e5b7491960df01a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-044b12081781ee7b9e9814feab1eb91fdf156b393d98255c7373c2abeeff9d8d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b07fb70427a2fff7ce409952db9c5a0e06cbc30d2757c05cf8ca188261bc6290 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-db928755ddf59cd4b1e2177896552acd7e7f1e594c3b12ca0e0f059b9b1ff63f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-104f3a81c0a86d377158d8b17532c9ca845eb1a4daa3f51e790dd79218841cd9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7b111216 | sha256-58ce3340061d5034ccd5b354a17d5d2c938d1fa06048414297a8a4bdaeb0af9e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7b111216 | sha256-58ce3340061d5034ccd5b354a17d5d2c938d1fa06048414297a8a4bdaeb0af9e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7b111216 | sha256-58ce3340061d5034ccd5b354a17d5d2c938d1fa06048414297a8a4bdaeb0af9e |
