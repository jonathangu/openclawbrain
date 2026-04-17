# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c07d86727cebfe369bee33466c114948863aec275c3915842adcc6210ff9f00`
- fixture hash: `sha256-8b304aedcdacbc80bf121116c28b99b2494a777738f36a03524f91c39297ceda`
- score hash: `sha256-0a30b0f858b93e9154c674f81876db919f9beec41d8af9d37dbe7acf772575ea`
- bundle hash: `sha256-98b1acf303644c89936ee4199e09ca187e47de5ff2442692fa8e260c2a5d1730`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-61200abce9d599bb6b2839cc09f35d3da44db5dcfc15754c19ad25f67b630577 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4400ef93fdc6f873a364811b2b88ce70877c15a41c14c0012e0d6e7b01110c4f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9e5e8ba54f305d3375b030c5b8aa5dd27008573918f431da07f834299a69d623 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-88ce8449e904b00d52aad59f8286571c36c0d39835da3efc2aae85fac2536597 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cdfd60df | sha256-ccc5d37ba82b4457d6dbe38465c12f3bb84b276f1c0b7fbbb1e44595a7ce287a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cdfd60df | sha256-1304bc00193f1e145928ed610f5fb6023515091f41984675e0cd5893502fdc80 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-6bc3d600 | sha256-6bedd8ddecfc90c4ee12297426939cb2cb575ea9105d067e51825677470e41e6 |
