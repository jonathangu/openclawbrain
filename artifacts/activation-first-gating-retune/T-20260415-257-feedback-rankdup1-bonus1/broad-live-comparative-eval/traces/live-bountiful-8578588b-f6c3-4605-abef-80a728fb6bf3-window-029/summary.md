# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-804025b59e9ccd56b6c987bbd4ac39904cd2a5df99a64be4a350dc631576367d`
- fixture hash: `sha256-e197071d3e574f8d14c2a018d3a6d553f258f6b2a618a40d8d6d516e0727c08e`
- score hash: `sha256-fc1172ba93a89cde7902f61bba55ed34e58a9d519f18ef756912c6abdb7a3a26`
- bundle hash: `sha256-275b473385aafd8d4bfe033ed8e68ac9a698988bfd9af4fb5afb5ed48ad59ed7`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9879dc85fee5e21c2aa8d0f905f0d82e912ca593acccb3a82008612064aa877f |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-e97fe4a3e2b0b25cffabeaf3fc78eae3e2b56a00099db00d1552b4844771377a |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-2046c3ed55d9abf4d580551664a4efb4e2ac2155ac55b80af274b7ededdd3da7 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2ca875dd85fde01cdb5749142e125f0b26a953abe67055104c42b9b835d5d116 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-dee2ccc5 | sha256-44f14985bb2b300cacafb7356aca6e6a4fec0b91ba0c8fa8ecd5485e15eec415 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-dee2ccc5 | sha256-44f14985bb2b300cacafb7356aca6e6a4fec0b91ba0c8fa8ecd5485e15eec415 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-dee2ccc5 | sha256-fc6dfd848cb41ae283e35582faf9dedf67fd482bdde4099ab8a3f5f319fddac1 |
