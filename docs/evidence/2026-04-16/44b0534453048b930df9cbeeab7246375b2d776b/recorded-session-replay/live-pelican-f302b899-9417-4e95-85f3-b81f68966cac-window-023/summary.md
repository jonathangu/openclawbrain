# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-858b0f43ef470e5eca2afe1da13983b8601f538afa969d1fec1c6c995e06b43b`
- fixture hash: `sha256-5c10bbce6c643d206da3406e24da637074a598036c8915b8615377d8cba78cd2`
- score hash: `sha256-20eadf63648b6aad1d42ebf301f7537c0551d7ad35b62f3105f5196e1027fca8`
- bundle hash: `sha256-eb9d648add5e91a666d5fca4ec5c1caaf32621795fff93ee8a336aac9599e889`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-17f51076699069dddd3d95709feed654148e93f103c223926ea5c4a4e2da537b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-329f152d9f5ca98c9edce360b72c0307a20966cfd8f078ccb9833f6e1f2e58ce |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-12a0e81d48022e219d92982232ccef41195fb02ce37e0228471050758ecc41b6 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-d69e82f7e2e4041839065f8aca128096fb054e1548d2e8d6d0a33383e6bba3fe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0694e693 | sha256-8bd665344830c2a3a76fef54f26a57598e7dc915d6705d695bdc08076255aafd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0694e693 | sha256-c8e4877afd6925b556fbbfcb5ad409d5fc9f2b32ec173ab592fee2c84db2ff5e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c7318368 | sha256-e7a3f5ae29a457fb6c1e9eb8532037c352557b3de663e712e7e4644689d843a3 |
