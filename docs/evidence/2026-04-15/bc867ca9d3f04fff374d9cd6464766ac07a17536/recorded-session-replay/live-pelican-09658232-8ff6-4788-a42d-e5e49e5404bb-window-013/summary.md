# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d380c2fd8773059a5893ffff2d380e86ca0f972a4140732a5832ea7865e5c2a`
- fixture hash: `sha256-55f386f545922fe7856a581e64b7fa651b1de1ce7956a55af05d3b2bdc86946b`
- score hash: `sha256-27219c0c5bfb9d27a99b748e9540a74e307b36fece09c81e948fd0410b46aa99`
- bundle hash: `sha256-1514cafe8b2f602ab4d05d24546184e8d4e362ae6a1d0c93b1c66daa0635522a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-63284d4d64db3291399ac8e17a28a524a22240af2c68e8497ef443766b42c4ca |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8884d4e2f70a027e3602c7347ccb90d1ba5d2b4fe8c7801c336f61192f122b5b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bef1276dd2d513aba277f3a71803e1a8cafbc707d9dc4ae22789bf54c9a946e4 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-37bddf134f86e385826024a1599cc2f1ba6872cfbf0a57b268c7db8d720fe4c6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6004f157 | sha256-118c1b9a6c47c65d56f0ae58292cee61d224073be22431d9a4c97f1bb4ed67af |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6004f157 | sha256-8d6615f4a2a773a39802281b8f4d3d04a0a0d15d3cbf0d06fe830f8dfe992b81 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-6004f157 | sha256-118c1b9a6c47c65d56f0ae58292cee61d224073be22431d9a4c97f1bb4ed67af |
