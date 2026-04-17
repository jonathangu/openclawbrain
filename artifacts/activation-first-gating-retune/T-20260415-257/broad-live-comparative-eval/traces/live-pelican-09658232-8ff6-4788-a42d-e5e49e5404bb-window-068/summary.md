# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d5b2e9dc9e67decfaf3c661978d40b3965c717607588db6b26b950194e4e66bc`
- fixture hash: `sha256-05d4de9ab3e3c70047bcf0e08acaa0f5e5762d96334a591c78e4a27669a8787c`
- score hash: `sha256-b5f1419b3bbff09025baa5fc25210d55cb7f3e252d9afd8fb71c79515548cc1e`
- bundle hash: `sha256-dc4dee81ccd896c5f158f2c89d2e753ef0fd448eaf6206fc4972780e27d07a11`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f366f9857f3448a88ae6a2a72b960ef51718ab24c799633a78bd702bc8b064bd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0fd2afbff493b04375a886e1440a5aa5a099fcc4aa9182fe7569d9f217e5882f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-88127542829006bf608e54ce1000f6a520e3113398b246aeba5d7cb34d42c655 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c3d1c521 | sha256-99284ef4a0a9877d8a222a8bfef01fefae205987b75042d03f24b8df3b7d7b23 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c3d1c521 | sha256-99284ef4a0a9877d8a222a8bfef01fefae205987b75042d03f24b8df3b7d7b23 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-56b7d260 | sha256-1f20330ac315e9a42c5f4c5dca9618db6bd57ea817f2e7b5f9846905c74e301e |
