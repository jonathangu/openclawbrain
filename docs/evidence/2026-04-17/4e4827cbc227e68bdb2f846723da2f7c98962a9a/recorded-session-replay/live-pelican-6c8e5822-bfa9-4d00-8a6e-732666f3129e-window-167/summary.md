# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9808f8fc2b34b9bdd2037973a4af69235ae13bd43fa03c06a9cd2c930faaaa29`
- fixture hash: `sha256-c59bcee0d7e5004e8699b7491ca609cee1baf21baa2824b5dbd8c966b365083b`
- score hash: `sha256-ebef1d4a467421cda7fd36a95be5269f3630f198bd099f37f09dff3d3fb40cbd`
- bundle hash: `sha256-6ef4c5dccb0e22e38653081d3402030a2fd89df5c240a042875408f711d935fc`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68b72eac2031bede9aa8770eaa1f000f5f4b3a15976311e630a954289393b0bd |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ffe0f0acce47141e4a894fe799c9b1a1800d57f87c7d976311e85204fcad9187 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-07840059ecb88343710a2190b93e04aee39a50e6a5694e8a87ef429ae705c360 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-3890e3e3720328a6da391c49cde3bfa69f56d9736e5dc211c6807034f4d01c04 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-4983c86a | sha256-52ad2068e5bd86cf1e4dfdc9743f5b47736e696b0baef5a3a23923723cc269b8 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-4983c86a | sha256-925be8ce5dcb818479aced8ecda5812bd3c01e56cd141a74c5802a81b7f53976 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-5278cf17 | sha256-ffa1751183d83c48dc621945bded4866b43dcb41da2988e50f4e255ccc2b2bd5 |
