# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1cc116d5a5a3e4268eee5081d6d597a83a2afaebb6c2529b01952ad2f45437c1`
- fixture hash: `sha256-ccd8a0f1240cc7f92941ab2c1ede0327e4ed0a420f6a51ec4c81e0437c7d59e2`
- score hash: `sha256-d50ee33f7f281d2e3c5f88973dd6536a14297f52bfc4cd79af09004ea5c1ef18`
- bundle hash: `sha256-19c65db42b680882fe7d6a5455de2141b4652f6071c966f0e92fb3c67775beed`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a97ee439ef356b4483f5735f34054ec24021480ea2dadec6ac22262eafbebd17 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-bd53a378ac8da8da71709709301b1f8206a9c6271e393cd3a2355db4f68129d9 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-3241ef19bd793bf4892eb6b79115a528fcd272d0b9f8977e00dae7eca96f6920 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d33cb8ba521611f74f06d69e67b3210c34c13105a571c8106fcf5abc68ed59c0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7a16ffff | sha256-6aa120b25fba85b12e6792740b47ee7d82b2bc4e65654f33a6e94472fda339f7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7a16ffff | sha256-05831bbc196ee22a02264936029aad20e9c3fd6b9ee72d052de80cbc62ae4632 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7a16ffff | sha256-f7a91e51c332b95ee4d150ddb0a798c3bbbd26d31932e92de424bf08e1a76cb7 |
