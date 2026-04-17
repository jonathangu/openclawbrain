# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1c2fee7fd4eb0c2720a3ba15050df8108cb036feca9a01fcd35c4b07aae7a9f5`
- fixture hash: `sha256-61f4419f55eaa7d0c0ca68a6f768711b70a4823f4e0fe058cff8927193ee8afc`
- score hash: `sha256-01628292f3b08c253491c6a6a96b455d82f6f8ef425a4c173b4da9f8d8e85e4b`
- bundle hash: `sha256-f335f8d6f4d502ebaa72e7d5924e9671eee78c67fb6c32a87f7d76bb086db12d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3dcc36ea1001cff13b10454b28af88c47e797eba5193d74b4990d61c1caa8eeb |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-55cdea6552d696c1f12f4d9f2def789d932c99e3b229c4eab04747fd2f9e1c14 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7531638de59d22a06daea3d2bdf5514c99a92febf6f243e02a12d3b607a3188b |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-e55fb9491992967500389589ea162c6554f6b198a8b284b82351adda4a07305c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-49fda424 | sha256-e7e7227bfb55e8d19cb2b487f51642279c073bca967f9671ceef4da6e903ad1d |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-49fda424 | sha256-c4403a79e086b716b589f683fc2f40652937099e28ae70ef031cb732c4e2c390 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-40f6b921 | sha256-07324ae8d0a2025744b844a6b880c025cd96faeddc88dcd6d0e762608edc8c64 |
