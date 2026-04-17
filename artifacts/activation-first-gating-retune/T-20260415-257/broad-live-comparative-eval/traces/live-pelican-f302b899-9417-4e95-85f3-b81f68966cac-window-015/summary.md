# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06c88cbd7b40857f6269dd03d5e04022f7a27c8c5e2a225bc79b2768cb90fdfd`
- fixture hash: `sha256-6d62bb5ab6456b9eec73e20f3d1a35ffc14e9452a4f4442f3b56ae134f63d27e`
- score hash: `sha256-f66e5d88b34c0a9cc1749883abe8cdcfff766d2c8297611e5324c903d1707f84`
- bundle hash: `sha256-fb895be9279f80981a0887fbd315eb3a68bec29d06d7fe50f0aa91596f7fb037`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4cd163af8984e87c72885a17249c9a84973c54f74e5363d963d16ae86c9b4e43 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e8bd4b83bcbcf783638940a67429591603be1e57ec360ffe7817487431bfa4e2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ad1b219fcda053e36c94041a0d8d3183f0c0c4214dc41ae2c4c7ccbad20e07bc |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-95f411e84591f420ca346e173a5f78ed0431edb32189e6effc7f4af5b3cbdf00 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3c4e249d | sha256-e09ba317a0ff489e088639b1568df3898c9a0db0aa6aed6bcdfbbff01f9911c4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3c4e249d | sha256-7a4d08592391099bebfb680c06d9ce9f9420649e1742fbf2b43c876cc986f7db |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-58c02a3c | sha256-c91238134ff545897c8c1d440df80bce0cef28df4de7bb1e9aaf9bac2769fb7e |
