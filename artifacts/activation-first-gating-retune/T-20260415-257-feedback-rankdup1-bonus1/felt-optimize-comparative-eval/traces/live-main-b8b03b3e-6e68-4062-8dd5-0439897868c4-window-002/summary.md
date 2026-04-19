# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68b2fa5dad518df8f3866150e5eb5ae2df4d40d0d1730a7b39326babb425756a`
- fixture hash: `sha256-9c558cf390c2d5519271f6ba91a97c5aab0727de8cfbeaa1362c2e39d2a00c20`
- score hash: `sha256-04356b15315a5e1c0f469395705fb95cb17cb4f660a7013d2915ed6707314d7f`
- bundle hash: `sha256-95d5ec9a538cb82200b1bc1b32425ccb1931e960bc8df361acd4e29a8e717073`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8709c6589a780862225e4afa90cbbf44ed4ef4f7b39772bdc54c0a9f8a33087 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-0bc031313a2dc851778248be042d411b8b3aaf04ef1c8be6b35f5dd81a977420 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-f8a37f3b30a9963797a5a9481ea9acde12f5a29d68b3ff9455a46bd848863a06 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-0c5860c4ebe299be9e243e6b7db8382c2a805bdc567095d4dc65f83402ab0993 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-bef75967 | sha256-a8bd52c413e149e74cf76c24082e307005a9fb65ded7231d21fa876fc66422c5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-bef75967 | sha256-cdbc2908e10e8a52116e3d3fa6f127d8f9cea7938fded7b58a1ae5ae475157b9 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-bef75967 | sha256-34dfdaa4e95d7fb4e4e50a9fb3a91ff0ed1a2a9f1ccf058c01677b725e318516 |
