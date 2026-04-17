# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f896b3d7889710642e066f81d9ef38f09a0375e7c4550a3de44bd42b8be0c728`
- fixture hash: `sha256-02b10f8d55f27089a7a2cdde95f78ea9472dddb1b95943a1431bda089a73cd5e`
- score hash: `sha256-bd69b2774f03a860e0e924e00794c084fefe8fcaa5affae35ab770aaf4f87a6f`
- bundle hash: `sha256-4eec7205840372f2e3e004eaa2c68d13fd64fa592576370d04131f01786976a2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6e466bcaf85a4528dcaf1f22f57a3cde69a22135dbdc628862617cea9e4f77f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2f966a24bd1bf25422987ee3c6bd14eb7ff3cc08c070c4ec715b0debfcef26d7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-35ad1e2aba6bba06bf2dc5b4add1e338e3410a36e3cd3fec44c0f8f395c960f4 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-881ddcf3e30ef70bec349a18717e840359478536b20da1dd2b92c677f8857284 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b32ea799 | sha256-a1ac047ebeadfc32f82c3badee9999c34db653cec036f2fef5e9f9779fb46dcf |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b32ea799 | sha256-bd62326c312441d90a03d1eae932a648b203d99680ae4d4d9a91974defac1acd |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-08155574 | sha256-6030c00225250fdc766bda39f63dddec8db5c7e1163544006cd3ad1fc0db0775 |
