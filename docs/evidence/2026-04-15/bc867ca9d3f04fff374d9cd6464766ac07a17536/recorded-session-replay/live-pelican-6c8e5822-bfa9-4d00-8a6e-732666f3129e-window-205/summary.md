# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f579c26265d9087760a95275a1ed5d3c29a7fa2a5745f0cd5985ac21a42da923`
- fixture hash: `sha256-344c0f8fa42bcaf494090e8fb4c4629c475783bea29bc527602ae9b6d23e9791`
- score hash: `sha256-0887d7835b84033b9faefb2d1df861744067f7b6f6523924780a2a3971bc7ccb`
- bundle hash: `sha256-c121d0accc6c7d78b366718aca5df15c586150dcc435443d7601cdd824186eb4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c8dbb3069efa791ffae92b155399c68bb15a7550a719f9b3772c99bdfa5fdc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-57757ab3d4bb16f375d5ae88da2e8a5c12baec080c1e5e3bb5650d152b7f63a6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5bef1d8cfafccdae34fe5a9d71d8c807518e763d648a53c0e3c9844c3b630b06 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-62301dec089bb4f6993e6253c807eee953fb29089156d6df5eda82a5faaba19d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9d608783 | sha256-9f9b64e427806cb40d7a8363d8394941532a667996fb1fbc522ec86af220d834 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9d608783 | sha256-542f72f74f325754be3d9ee9c0d073ddffeabf222c64b54d2aaf3d415c81dac2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9d608783 | sha256-9f9b64e427806cb40d7a8363d8394941532a667996fb1fbc522ec86af220d834 |
