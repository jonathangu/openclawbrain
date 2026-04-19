# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ad0e2983c4f13057addac140e7ec01136b02517807f2983a2a8218c39f77ac60`
- fixture hash: `sha256-5465f326b57b932fce5d721740c2e94691b72cdbb86ecc6d3f5feebd376f974d`
- score hash: `sha256-8cd94db0e06a5069a43c8ae99ea226b2abcd52d69a707aed017efcadad24e98b`
- bundle hash: `sha256-6c01b851c4c35cb055df496f017dac5996893b15074bea951721f202b49d4565`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-11232886674dc9e702bf2807ba0bdbf15e55d8f77564bf0f29c440dc177c94e5 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-d9b61ddc0d8f7344ed272003c65136f9a3d5d6471a90942abf32a9443a3df795 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-0692b0352b0bf49bda506edbb252f765fddc3bc0a4e51b65d39639b092ff1477 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-ead1b13a285f82b69f585d0127f1278a9a260092356be5b1240140d25df03e26 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-509750c0 | sha256-0ef7544ebaf1bd4d2e1c06327df13f2302926d4236769b081b23c6ecc62ad905 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-509750c0 | sha256-0ef7544ebaf1bd4d2e1c06327df13f2302926d4236769b081b23c6ecc62ad905 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-509750c0 | sha256-1bb1ce198900a36784420d7116789563d7219b9ec716d257cb369c8d73c25d6a |
