# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8226e38f2d583af41a4327f3b8df4e5b434ae18ebbdb89d67531a4a854359a44`
- fixture hash: `sha256-e3733e9aa09beb01fe43936408b2069d985913ff1742752483045d9debec0829`
- score hash: `sha256-dd59b7b17b7a029b9bbd13d19aa2e019a7cbd48d1141b0e595a4e922090b91f1`
- bundle hash: `sha256-d686a167f8765c35d759fc867263973c0f24116f6abf81abd03e190c4bdaeaef`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f1f95cc8e218fff5d5905cf899fc04d3d3c62a98c1d684ae5ae4dffaa6f7bd10 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1eec31bed7e07adfd0ccf436b3a875f82db9b37455d3f274689cd8f47cbd5fdf |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d195565e9778148bde2f1b6a701a3f899ae3a427a1c09f9210fffa2cbe4b8be9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3de13008b5f2923260085a8f88dec255977d89cca957b9a82e2e8bcdafc1a6ea |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bdbd45ee | sha256-f254b184432f5f242cc91496b06d6128f2d9cbc50986aa2282d3f9f9a6aaa404 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bdbd45ee | sha256-d098b3b4677d5b2ac734bc279518304e5055f04a7881eb984e24e025e5b4bdd3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-66ef96fb | sha256-a334f387c8ff19acd2b5af726d39ef1596e59271433b838ef883642c29c02a85 |
