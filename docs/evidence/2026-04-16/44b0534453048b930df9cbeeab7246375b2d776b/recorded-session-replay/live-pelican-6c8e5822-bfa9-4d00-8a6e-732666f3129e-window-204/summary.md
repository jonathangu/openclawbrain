# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c07d86727cebfe369bee33466c114948863aec275c3915842adcc6210ff9f00`
- fixture hash: `sha256-8b304aedcdacbc80bf121116c28b99b2494a777738f36a03524f91c39297ceda`
- score hash: `sha256-5d6960dcde9cb5ca3f54aea74e26e0da540b94af68cb8420f02b7f4ac791f51e`
- bundle hash: `sha256-065c59a770f1c68b88d40ae88f96d9a42fbd243c2c7e339f09e7794778f3ed72`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-61b81fc16cc50aa61d66a1fa83957a49462e26dec7cb306faa8592e653a25d34 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-98a924ec16be7f4dd144f2bd5f996ddc70137af668ed87f5e55ac03b0e30ca07 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-a98d713ed2371ad1c4cc6f53d822b7808eb8f15d26066890cd052b52785757f3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a5e4aec8 | sha256-357b6499a28d21ef519e0ec53f9e4463d44350da28c6c82cba414fee993d431e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a5e4aec8 | sha256-1d6383210a94fb7572beeea3f3dd0ffaa624b2d6e55178630d2a5b19632e59d3 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-43ab23e9 | sha256-9bf236fd86258557b44f6812fe36e7ba76ae6a75f64d8b530d56d90a438b5b4d |
