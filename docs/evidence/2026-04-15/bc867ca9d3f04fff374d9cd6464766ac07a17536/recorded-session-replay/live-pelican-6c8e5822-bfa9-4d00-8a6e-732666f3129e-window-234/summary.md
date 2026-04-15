# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80d3477f10050166bf08a79ad115cc0623875c77edbf3489b3449d2e77618193`
- fixture hash: `sha256-550621052f6f6f4dedd32e7dd1966df3bdae13f0842e74ffdcfed29aa308dfb9`
- score hash: `sha256-aad7b8c15eccfd642e60a9254dbb658adfa8db3879f1d1f39f85e15a1c799a24`
- bundle hash: `sha256-cd40f5d92ed3d4882f01801c4eb4d72b7dd1ba95ae069b3049ee00daed72ecf8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b053d8d2940510defb6223852e2cebf21b6ccd631a727caf5859e48b2c5a0baf |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e4a152c625dc3e6feec911f5ebf02f4fb7dc633b820819f90625562a35cbafb1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-304a008e3f63a8a572c1dc6fe226ebe6897e405a69bb4b6b396b35c195e3a1a7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-51b121b6b7e68024f85e981e1fa8623f0b9c75f308d8674901491c14de10e7d9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b67dbe7a | sha256-2dbe35a033bd5c19969b7d732d3609e340c29164119db84de3e6628b808ee73c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b67dbe7a | sha256-442a3d2fb47b0118c30887a358a1d2809129b3b696dae7074223e077bd70f1e2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b67dbe7a | sha256-2dbe35a033bd5c19969b7d732d3609e340c29164119db84de3e6628b808ee73c |
