# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92040fed208ea65585f475c24b64fc03720a2a86a8c84eeb65240f8ebda78b47`
- fixture hash: `sha256-e789097b492b370a3cb207f40a7a3a195c61c7549ef2c7d39a0e569e0dd15633`
- score hash: `sha256-4f46f927b60cd738a37d5b9832e2e581b47f46d7366ddfa7fe716c006b98076b`
- bundle hash: `sha256-6e2f536b4bc1e1d32969217e3c47ba7822d532bc7ff88079945712ac2ef06798`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-648eaa1db4d7048b0d51fcc33cb635f35068c6efd52393b35ef8355c224bb749 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ff25f721c6eeb24cf52eb6a977360c7ba2affecd01812135ccbb9ba67b124462 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-620500125373f066df8389bdb12905395d7ea83aa81785c043a974d47036451e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8c58bd2fcbd128761c9f8f865096d2e1b0bfd56cfdf5d8bb70d8a927e2c841b8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8b7359c9 | sha256-d5cdb34aa8bdb2feffec77083b1863b97b5bf7958659599d0d3338e68e90a030 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8b7359c9 | sha256-7da4699d83f402cc1c8879ac999f9ae51d493808e8bb8e2271b10ec38d9e7bb2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8b7359c9 | sha256-d5cdb34aa8bdb2feffec77083b1863b97b5bf7958659599d0d3338e68e90a030 |
