# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f94013a0a094ba3fb5146fbbc0c01b19cd73626385a314188a4301a7c00be04`
- fixture hash: `sha256-113ed14948559f61f0991314db5ab7b153e15f743e52adfd432a03d575e47935`
- score hash: `sha256-1349e3057bf27ba0d4bad31cbd3878c9f74800db2ae2f3f7f8418abd8264c898`
- bundle hash: `sha256-2be5417cf8c5da950ee4fa09602552a0bc77bc2442a906c692ef968559375d00`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b0dc5c5b8bcb0e2bae32d52cdbc81c4d2b373af818d11c1ab51e1555491f474b |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-0d00521eb57fc9a130b4510f0af6402411716825db1111f5a18c9481469bb77d |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-2a55da527d47fa6b211933d40e0089c2406798741aa38404d394a4af97feb7bd |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5df2878a0d9d416953575ce96e7f155caf4c040a255f2eae03404f4556a3c939 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-984b8f38 | sha256-d0dbdc9ed9dc475efa000cf06cbf06fb68dc0b5473050f89ed46332f563bdca4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-984b8f38 | sha256-7ecd3317319a9367a0638ece772f00ccc941839883212884417d3de0b847fd9b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-984b8f38 | sha256-d0dbdc9ed9dc475efa000cf06cbf06fb68dc0b5473050f89ed46332f563bdca4 |
