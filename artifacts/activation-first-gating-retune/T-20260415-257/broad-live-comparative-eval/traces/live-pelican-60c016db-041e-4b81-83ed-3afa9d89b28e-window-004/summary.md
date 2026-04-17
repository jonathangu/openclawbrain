# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98f498b917834ee9c0a78d5b62a338d5c94ab2df87cb501ae8615cf42d07619a`
- fixture hash: `sha256-54eb8df766feda2c6211a7171b884e66a0008ed710f7d28bcb6341bc861e92a9`
- score hash: `sha256-acc1b3920323171912317ca0919a346c09517befd98392c9eded08e441f5df61`
- bundle hash: `sha256-c9753228350fa6b9113d9f69c33168d33a8a35bdf1f5ef8882a2ef73298820ca`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6670af2aca1f1e71cbf3c0f145ce7f96dddb89bb0330719aa7609642a8108f9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-500ca069102aa7ee8aaf3b83d94a318b938b99b06055a92d0758278992b0414f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-77d92058f257dbad65c1030d961e83d83d782e2e22a93c0d0e6876f9fee16bbf |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a643c29e2b36d5fad2d5fbe243207acefb54a30a4d33ab4fe20ce52b3b6d462c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e5b59296 | sha256-75d75867f013a470a81e6070ded5d3770fa1a81d728c6424dbc83792a09a003e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e5b59296 | sha256-c28a43a6e24595a6f8faffa4af88ac6c5a7e0a01660a03f17f63ab0425924300 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a3d0224b | sha256-8f3a89d0d9fd30fefbf9351bf877005af0a442106ae65ece9d993d6494fdc90d |
