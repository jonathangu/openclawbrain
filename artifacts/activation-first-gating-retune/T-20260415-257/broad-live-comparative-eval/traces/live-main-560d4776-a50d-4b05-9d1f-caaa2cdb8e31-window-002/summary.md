# Recorded Session Replay Proof Bundle

- trace id: `live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-035806e6f9bcb3753f58456b70d56dc8a01f4abf60114aeaf359384806f6c24b`
- fixture hash: `sha256-74bbbcc2ba3e23b87dadc56cde438b46daa30c3743245ccd0b40d24de1249370`
- score hash: `sha256-957eaf3fbab64f021842a509b68d54399b6a277b81927cf4c7f08eb5a7ebe2d3`
- bundle hash: `sha256-dda0bc78deb581b3fac87a281dfad82bdeff23704d1a94d43eef6cb3c2a464de`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | vector_only | 80 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 4/12
- phrase hit rate: 0.333333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54ce8361598e1b4080ba115badc91e906441ece2076bc91bc1f9f28df2706034 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-aaaec0ad942066322a218ef005de122fbe503eabc3315a87cccbd5a71fe4ea9d |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-78a91008299b5508cf3a7e42bf55d8d513c18951d7a89e5d156b0d6d1f1d02f4 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7c488f87b58f64fe7af7f3e54109fd78e5772b321f267b3e919ea12a417a0d5e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-83e9262b | sha256-61d7693f6422d8d8414b57513725d46d3662070847fc80638f5f9ef8b61537b6 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-83e9262b | sha256-b702f9473576ecfc6c37d75e878987bb0a1e3d7a5f7aa9d1ccd215ceabfe30cc |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-31a9ff4a | sha256-da9b0c41567fe1991d6026a316072a3abbeacb7fb03dc4eb296e0c9b9b848c50 |
