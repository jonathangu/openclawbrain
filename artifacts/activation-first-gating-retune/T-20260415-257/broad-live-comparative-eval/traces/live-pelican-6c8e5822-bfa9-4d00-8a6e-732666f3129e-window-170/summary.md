# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b236d1de0cb5cdb5c6745d8cf8165eea05f61fe4f12fe91030959e1a1f0a9ef6`
- fixture hash: `sha256-2918c2c3bf776980cb54310652408dcd4b80904c74dc802c02149421011a5050`
- score hash: `sha256-fdd89d7e7ec9fe6061aff381e1bfb0e3b0c229eda2f90afebcd154dd1b907407`
- bundle hash: `sha256-d815f246a131f4aba8e8569acdf77892c60548de949d88bd7a3ac200a1cf81f4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b752caeb990c3acabff60b3183401c5659a9fe06fb13d30bacaaff23a3d4f453 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6eb39bb2b5761045bd59bad84ceaa41048712c05cf5b381f8e4e49a179b09304 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-45c853b27fddcecb743a55f6ee11850a5cd608aa3dea720b4e4702de016f13bd |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4f51323ebdc1e963d6b1dc14e93c62694abfc5e08a2179a23f0524e87f21f085 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4e46e756 | sha256-be99590b09f74fab49bc1ca8268b901e109c50d60477e013f9ef7f7a47e69d4d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4e46e756 | sha256-218a82ddb96aa140398fd42702f0a72e4f3f1183848d728f108397e2c42f5730 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-40fd27db | sha256-156c9074739b492d62d77e63a92d2ad2e7731fd195bfec2d8af322086b2abc2a |
