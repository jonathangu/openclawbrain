# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d5b2e9dc9e67decfaf3c661978d40b3965c717607588db6b26b950194e4e66bc`
- fixture hash: `sha256-05d4de9ab3e3c70047bcf0e08acaa0f5e5762d96334a591c78e4a27669a8787c`
- score hash: `sha256-d959738002e1d29fcee2c33a4f78a674aad5aa2693dd0749a4790a2cb601d717`
- bundle hash: `sha256-5fc53a8709b2f69f3b128ea8e4e1dd2965c602c5cca471775d00838b366ffa0a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-044b12081781ee7b9e9814feab1eb91fdf156b393d98255c7373c2abeeff9d8d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7a9fb3426020f0a37a4aed7aee680b5f8a997c02cb28d1b174365f3ac8a238c3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec4cbe09250dd9b8f23c6f2bb2d24aa583f2a899056e88ffa8b56dceeb05aa12 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-13de872f0bf628f2306941899369c2b231109984e917a8e0b1e2399463bb8ee7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ecc6c2f8 | sha256-1fc7d12c47687fc094fff00f6186ac38efc472bd9e0b00b1d3ba4facfc5d8ee3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ecc6c2f8 | sha256-1fc7d12c47687fc094fff00f6186ac38efc472bd9e0b00b1d3ba4facfc5d8ee3 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7facd037 | sha256-b25f46300c882625ba3bbcf74340b07f86d8534a1db3d43b0ae8b186e2068190 |
