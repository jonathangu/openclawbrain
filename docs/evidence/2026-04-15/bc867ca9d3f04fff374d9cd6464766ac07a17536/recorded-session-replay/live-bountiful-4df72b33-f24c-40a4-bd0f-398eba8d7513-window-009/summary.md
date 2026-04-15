# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-909da7829d18c7b4060630b0679e4b4f9d3623ad878ab660055447ba4071489f`
- fixture hash: `sha256-d6f02df1f7fd44472c7a5dc57cc2a6eccb52d15bdd1b99973bade05111901191`
- score hash: `sha256-60de269c423bdaded56219365448887f513b587d6974b5f0aad226195ed68f2b`
- bundle hash: `sha256-600ed322cfd84f6c96be9249398f85d43b6338d30d92d4827f5c31b4999bf1e4`

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
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-67d503fa6cc9652ac016339af3a8c1038900f3a66d390047a5f53eeceb6dc18e |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e24c8d27fcc72998920730f5717e305d646d36b2c5fbd9f9bdd676dd3c3b0a8 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6f9e92089ec20d8bd54d7f1e2ae3aece8d8254e6946f3ed1ee7029c2b86968c |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-4771bfcbe29ac1080a7277f57bcce1b35f32b79da6ffeecb9b8abb695aeea16b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d0511df0 | sha256-bd4cee0b23fca76097145ca59ae27c2fa4379e332e750703bda8608dcc8354b6 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d0511df0 | sha256-818fd91575d6ed58c3e2d4a5c9755425157178020d2fd93ef790db045e708dde |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-d0511df0 | sha256-bd4cee0b23fca76097145ca59ae27c2fa4379e332e750703bda8608dcc8354b6 |
