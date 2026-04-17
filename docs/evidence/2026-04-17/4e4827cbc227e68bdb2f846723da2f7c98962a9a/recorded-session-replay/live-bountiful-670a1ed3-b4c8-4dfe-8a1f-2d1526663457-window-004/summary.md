# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cf10cae0f36c648f32a6c50dd8217f4092591b4c33fa07516441723d723d101`
- fixture hash: `sha256-74ccb99a45cbecfbd0675ba926480f518b6d9257f4cbecb8a7eccfb5e3bc826f`
- score hash: `sha256-0c9d81f2b7cbcbde3d4764c661a29cb2068edcfdec5a710ef1851f5cd8cfc1f8`
- bundle hash: `sha256-ac0f70288a3a279599160ac77b11b858db6c97b80eccdbe2112a894256b1473f`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-deab881832958a8bd935ee6c81daddd68f45a0ca219749d213e1a30ab0bb8c14 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0dc2ee53f46d072dcf60fedea8f3329e2e81978723f7124c36906530d6a8be46 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7bfa6f78bfa45bf8ff50866d6b00f9f94a25cecbfcdf91136dd3faa25c9b02a7 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-465d41f838973e10ce2a05fa2c9aa5190e7582f2940c67f2301cff89714230c1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-56c10cd9 | sha256-63bfa242acaf36977a5a0d77736b1a8c106b5068abc138ecee9726add6a1995a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-56c10cd9 | sha256-beec70ba440e28b3a896ab44eaf4702db805af0c7b1205a57be9b38d70111141 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-37909bca | sha256-4d14bfa6828d20296035d8cfb8c6f83b44f34b754980c8df0c25f3ce8198a276 |
