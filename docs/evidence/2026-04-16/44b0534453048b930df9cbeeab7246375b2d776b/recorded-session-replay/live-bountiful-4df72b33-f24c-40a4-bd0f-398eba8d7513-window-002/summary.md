# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-25e681dde9bf99a5066e3fd272c254e137908dd8248f9cd30c28377b5642eb80`
- fixture hash: `sha256-118dd0d43d47e09d3e0fb14557115fffb91ecc9b2c9362bf193950d5af577035`
- score hash: `sha256-9be37e7fe4f2ee974d323e99c7017606e1e57e409a2a380cae0d4485e4214dd4`
- bundle hash: `sha256-db3ee089aa82dc19d1b2d1ab921ceb2d7f14a891f47c44aa0b584a40def52010`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
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
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-b303f29564dbca934435f85bb92a8e88524aa1c601798629b1c393abe799f6bd |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-a96a2ed0de1a4f3b521f6d744dbdac4300ffb741c062416f1be51695ad54d7c5 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-af3ee1d39466f25f83d043dfbe238ecfd46ab9b957321e8744590ec97ece13cf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-9f426455 | sha256-bcd41def9d6f5baf763016ae3e5bc052bcb1102059ea8b04f160e17b95f6a5f7 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-9f426455 | sha256-bcd41def9d6f5baf763016ae3e5bc052bcb1102059ea8b04f160e17b95f6a5f7 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-5b6fe070 | sha256-fae61a8293003d43438d36bea69d03e359d1f46d4a58a7838af7d0d428f5c710 |
