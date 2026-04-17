# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eebf5a156737b8a1b13583833520fd34225ae0f30b4afb05ce10671f54ea2108`
- fixture hash: `sha256-63d1e3dd69e143127b58a78b17f85b8f588fdddd25950ce30e59877032c4d44a`
- score hash: `sha256-58fe96e5f1f0acbc7405215484e4c7bfb9e1e88c9684c256ff84d6f057f54b7f`
- bundle hash: `sha256-e89b0e780881eb09eb4c86e492fd451c220c8a12f3fa8c740d89bd6da796ce4e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-798f7833fec1b062f8a2789c97b9a979ee4a90e5e78bf32289929e17459a82fb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5ac37e3725bd2f94edba81cf6e2a0932973b5e492aa68a60cd2cc595ca1d4def |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-94ea1a711997514c1e52b1ec44e13fa30854a879cd7ab507808855fe97163b4a |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-2c7ee854d4c5b63d30012ee128f2490ce855f8818d17ce185d70a6be1decc997 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d0ced2a5 | sha256-2c4d28f70a74eed56a5cccb600bb5cbb8fa73667791e6ff9cf09a4992631bb66 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d0ced2a5 | sha256-6e7b28c37a2690b9f02e1a0d1f83b429b0a5c9dec9434f79d72a183ce4608a22 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-76fd80da | sha256-1937119b91c871ef4e97778c5497462ff17efe733c0647babbc50d376255bc9b |
