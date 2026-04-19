# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-25e681dde9bf99a5066e3fd272c254e137908dd8248f9cd30c28377b5642eb80`
- fixture hash: `sha256-118dd0d43d47e09d3e0fb14557115fffb91ecc9b2c9362bf193950d5af577035`
- score hash: `sha256-5321d0771cdb89f9c3d0fbddbeeb613dd11a5fe0a97b1adafe8afcdbd1641be7`
- bundle hash: `sha256-b3ed96e7ece752936c8e49f32141710e2b229b0a2f64eccd14e00d31051cbd59`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 80 |
| 2 | vector_only | 80 |
| 3 | graph_prior_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 5/12
- phrase hit rate: 0.416667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-053f07f407b9f0886975eb3e4d95aa7c39bed9e8cf96e6716ec7a7f71273ccd2 |
| vector_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-2ece3482375159a5e1b54ffb67f8d013ca7dc48b1675a2b4d4d726017342fe0b |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-0aa6760fde10b689f4be8214f9f8d0558d2b50963e2058578c4fb185fd67ceee |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-efc3de55ec44ec228d48c205cfd4cbaf96587aef641825d21622839aa4dbe505 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-3389593a | sha256-8772ee32a246120fa4bf6da5755b60e5bc3e6e6b74570ca6f79b325acb153c78 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-3389593a | sha256-aa5ca5400309d87117fa9cd691a254c95e8c5103d5d160434b850bfc96ab013a |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-3389593a | sha256-8772ee32a246120fa4bf6da5755b60e5bc3e6e6b74570ca6f79b325acb153c78 |
