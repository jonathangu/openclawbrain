# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cf10cae0f36c648f32a6c50dd8217f4092591b4c33fa07516441723d723d101`
- fixture hash: `sha256-74ccb99a45cbecfbd0675ba926480f518b6d9257f4cbecb8a7eccfb5e3bc826f`
- score hash: `sha256-0d30e035ddf0a6c7ec444febe5fa458a7bd58894c66b51c95e7eb736bcbdafad`
- bundle hash: `sha256-16d015ecf6523e8a577dd63e0dbf4b91657ee0d3f8715006a4a6541f09f0455e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-deab881832958a8bd935ee6c81daddd68f45a0ca219749d213e1a30ab0bb8c14 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0d61ec58ad5906399cdc6eccc5a5251ed442c4a567ba5ce973c2dcc4f863e38f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f92fae3b020543a98fa4670df3da2ea97ab868919eb6a1446abceaabf91184a8 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-5700e5bea9b86571fa00740426408df7b110dde72b4a1edf19392a74b9861f58 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9e3d3859 | sha256-70fff09392161c3c96eda66127ff3c750d6a1a28f48eaef50743fec09f82c536 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9e3d3859 | sha256-473e4567defb48a0bc25c6695e48638878612ded6f56c1eccd42ac2e511208a9 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-7f0cc74a | sha256-c16033e5938003df59bd4a3716dacb6fce3fbc163e89d31f28c859de4ff70063 |
