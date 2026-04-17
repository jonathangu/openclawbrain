# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8226e38f2d583af41a4327f3b8df4e5b434ae18ebbdb89d67531a4a854359a44`
- fixture hash: `sha256-e3733e9aa09beb01fe43936408b2069d985913ff1742752483045d9debec0829`
- score hash: `sha256-7b9b1368c9f2b12e5ceffbc4f97816012a251d34e6354f59be8faa9f1fbb46ab`
- bundle hash: `sha256-c48b4f5f29ffb86abd85ffe20be1ccf496ab8d71780d849aecfa3f3f9ca6cbb4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f1f95cc8e218fff5d5905cf899fc04d3d3c62a98c1d684ae5ae4dffaa6f7bd10 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6618bf1fcffe9c74e81c27dcdae246028858d9abd6faccffa0bb3131881f91a9 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0a922a8fdecee821e8cbcc502a3fcaeccde8dbf1398f2ef6d1620dd9335275bb |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-93d206cfd64652d2a97c7edf3b09354ae99219974209d870a4c531642c0ae8c4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8ed58579 | sha256-d2991302d8fcc40579b381e123e1ee88decb72594fd4cc222ea9791f74ef0919 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8ed58579 | sha256-bcbe9d9aadc26811bdf56b09336167dca062cf56e6c9606ee454f8fa52b3d3f4 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3807d686 | sha256-1f306926f1f8e851efa1235872473c8833856d010472c74d933f40196a072ceb |
