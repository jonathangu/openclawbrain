# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adc21e40f3c3bdc2111e458183ef292b9fdba4cc9072a5e4575150e3a25e7599`
- fixture hash: `sha256-82594518eb539bcd92075469119fdd7049793972cdce0d3d047ffdabe9e539b7`
- score hash: `sha256-89a587bc42101ad9e9560050dbc9b318f1cfeda7f7bd7e1ed5a425f250502d4a`
- bundle hash: `sha256-f6f84d3e218ac0082aecd08df9082f507fa1fe4231551dfc204c82d32b0053b8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca48aec6e03fc6ebf10d02ee2af1729bb6ff692653b0f22ac3e3b10f844865d0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7aea90ed501325b5918ca040858fd228a5c6697241d84dd28770dc5905e35465 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-240789dc159e528b48eca1d01f157bff7f33871667302d5224ec265bede00f0f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-dbcd89b2aaf5b680a00c8a455477e253c74611eae2522f9f6b1e34da0d6b98bd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5eaa8db0 | sha256-2ee96a87a90b436afd0a8094ad0b087d1d1a4a4057f25b180c698b79a570776a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5eaa8db0 | sha256-b2f1af633d47738bae6cf68e9d475a7631b982b7aa9a3f40e8ad86d4b71fa9b1 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-8992e393 | sha256-3e114b031aaf5f7e3f208874576436945fe8c5cfad5a9292c4a4e143a1de6b2c |
