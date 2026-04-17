# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d0b5d294f5bcac07c81e1e9b7fbd08fc02ac60b4a0afd2bd2ad3564216748c02`
- fixture hash: `sha256-10c2797f7098132dfd19e74471fe861e4fd990acaf92ba667dc395a281a0c32a`
- score hash: `sha256-81f41d70c606216b67fc10fb19a564c9f43c2851adaaf9b2433f125f0b4abc7b`
- bundle hash: `sha256-880bbed77f2a622d7b0199fba4e8299bd12de4256fcb478fc9171b32cc652d3c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eb29fb5cb8fb01ea6e12d04715c0ac66ad31c35de2501ab2ab9a23569a1d387a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5b5486e280ba17bd5a7253d1eacce4466e9b2de032927d8b5a0ea5683356b718 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8cb1880ae625303cfc95690b8af8e89e4005e07f02e1b5f47fef60534260a11a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f51c6379d117ea0d17a03d56c5c6a6828eb696f0f1ffe217d3da6d4f4f400e4c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b4c30fae | sha256-18fc728b05cd4a003885aef4f386acd1d6ab64527ba9d876b628c7c3f17e8481 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b4c30fae | sha256-1b1b39b243d8f87d3b01a45e23ace6e4667cbd458ddd67752a3ae207364f61b9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-0a000997 | sha256-3083a0594bf7d80c1b4e447e76f26b8b7892718c537f465f5e085ba875b0069e |
