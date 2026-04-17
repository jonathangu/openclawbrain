# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d0b5d294f5bcac07c81e1e9b7fbd08fc02ac60b4a0afd2bd2ad3564216748c02`
- fixture hash: `sha256-10c2797f7098132dfd19e74471fe861e4fd990acaf92ba667dc395a281a0c32a`
- score hash: `sha256-a57f49f4d539974e74a6a5c6ffa108f505f46c3648a951ec9e335f4d21bd7f95`
- bundle hash: `sha256-b013bdb12a2c73422f09151f2729fc3df9b1ef7f09565f8dee6da7d0bd74769c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eb29fb5cb8fb01ea6e12d04715c0ac66ad31c35de2501ab2ab9a23569a1d387a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e84d5ccec34ae02bab735d8576dd3b90825d45db1bc6082def9668a36495d9a2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1583fed77c911471fe47b4892048e679497bd405688feb34b8daa8bb55baf5a5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c9074f2df183036a5a26da715669787d6aa7d411a0d0c6ba9a95fd768a5d68d6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0ef07c70 | sha256-fa836044bb176b30975117c93d276de55c60d8a24ee8a1f2ec2cad5f3ace7d04 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0ef07c70 | sha256-f35bbd2363a3383d442a1915af1644b725a39846d9b36a5d4feecca4ba9b4329 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-642d7659 | sha256-6e1c51bbbaf0ca1a35b5a1e8ef4978e37e3fe81a8d443cb6eef0aef6d845b54d |
