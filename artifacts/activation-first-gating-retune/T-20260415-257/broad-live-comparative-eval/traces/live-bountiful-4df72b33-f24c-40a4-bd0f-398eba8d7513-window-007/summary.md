# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b0963b5e18b40bd244437e7e0701feca11058e18371c7c7830706c53a28f15f0`
- fixture hash: `sha256-ad1fe96e9866fc7227d860e828f71679e46d996ff90526f19b5279748e32ad9b`
- score hash: `sha256-4d64791b8f1d19f92c9f5d1a34688c95fd0be6511a64b23650b7378870702ddd`
- bundle hash: `sha256-4316b7392d627cedcc471e43ff586cee5a3c2e05a9b640649487019007d9355d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-63768ebf632f25773108234ec4f8850307fca437412854c0aa69b01f97c1eac8 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6c31224a8bdc5bf67639eb7ae279bafda7d4cd8df06fa827e1c16491b12f022 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-f14225561b38bbfbf7af9541f3905c39af45b86e143110cf5c017b3ba6fe0b70 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8fcf5bfa285f0e2028ddae132cf69765bebadd19bf6c2c4b10805d2725481aea |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-afd1d984 | sha256-2074b10155e0988cf199f59ebce064c4a3364209769c9cdaa4b880b4004fa8b1 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-afd1d984 | sha256-e0ebc8a4e22ff68884a58768411e0ff39a41dcc3f8127f1a4071bb2a415572b5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-612e22eb | sha256-100d80794fd8ae6edc93603406ff18eac607452d8ba65053963453413b846c0f |
