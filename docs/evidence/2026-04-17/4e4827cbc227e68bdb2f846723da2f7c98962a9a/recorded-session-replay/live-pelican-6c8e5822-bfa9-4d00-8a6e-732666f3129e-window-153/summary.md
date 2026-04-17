# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffcf94e58297f053bce53168278403d4ee13aef69fa248575deb3926c6117a0c`
- fixture hash: `sha256-69a203d1bba5e9efdb04c3d2b5eac78a0fd9782e268e61f935bcf93878b096ff`
- score hash: `sha256-ed53de3dc1d4ce8d11f0d00ba06682f55587fc7e700fe180f1c0c4ced5f83642`
- bundle hash: `sha256-d0c469ab1dd4ee0e6ca3f7c53c50e12bd85c608ef6ad60425fecb770cf91b0c7`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d38606df0a6b5cc6fe27f296186c09efc80579f0832811cf6184d8073ca5500a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6ffebe868d1312af906ba61e6c33700c7b9bcbcc5ca29c765390efc659abe3ba |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ae48ed39caaa23b187cd2737eabf0cda2fd2d144b428cbd8b104dad106a4ce12 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7827901563bdbfe89c79766068cba06937119339bc18c68080591140a0783818 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2900e284 | sha256-0a3682777f11f37bdcb46aff503f699d2de0fb933a8e33e512bc305e26c892d2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2900e284 | sha256-282c6b6a5c604ae9781a86dc9209e10d52f7863e65a49ce8e88e4c8ef13de84c |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-31b7c2e5 | sha256-7e42ab78013d70ae33ba72002b4b39040ebb3e86bc5b8161402699c79911c68c |
