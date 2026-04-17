# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fd4a73ef0679d3bd5e8a41ecf8528eaf1056f459a2933d6bce7a274e1da6704d`
- fixture hash: `sha256-cdbe046df5ba47eb867d34f32f856111ce7f2bac423e41168b29efa3bc680b6e`
- score hash: `sha256-dc0fbeb352ef7fbe48684511a74b6831e0d7f5e3c75d06a7f902fb5f238f36e2`
- bundle hash: `sha256-154059aec0b5c1b1e59fa3a058a4babc32bb5cdeb60c689cf405fa961a786543`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-195c8562b43d566f299d3b4d568af19c059fadcd5ad0dc52c1779f850a2eeca5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ce746d5c7162b323d5e5a83eced6651bcde52e1ece20528671f361d3b8e9f250 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e65b88d26cc819ef57c9d5568a07f1d453a71a8c8facb6facd174098306f8c28 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-92f2da3d560fb7a2e64613d70defbca629fb7b145e9377661c7c0012cf814dbd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-64b20551 | sha256-dfb3546892e4d41f64f6e0aaf7be903def058e297f26b265ee54230cd3d468a3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-64b20551 | sha256-5231f0c1b2e5f1aa00b1b655305490f5ca8efcb56136f3b06a97213693b881ad |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-a143cb24 | sha256-6e1c6aab72a8b360bfe5d8015788f33e319f642965ef82d801d8f0777a29620c |
