# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7c9bbe1bc32703bdc0ba57cd7c2e5ba0147d232db874dca18f8d1c93a644936d`
- fixture hash: `sha256-1632c273e7fcb25c5de9fdb5adf5c07fcc4c43677737f0e63cd97217f3d6d9e5`
- score hash: `sha256-6459907dc58438df340a358f49734f7b8e33ec89b6ad3a942a68634c4588d3dc`
- bundle hash: `sha256-6b1db0633ff3921d5189c9cb7ddc0587eb184326cf4169cd7693c12034e648f6`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-80e9af933d3a18c2836442131236b812d1fdf8db3bb96c2fc77c951fce5a2ed4 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b337280fb61aa46c177e3a56c437d5e31a4739a8d6c7cf03b7eeeb1e250e3bea |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-dc5177a51fc9e3bb50bc56c6d4782ec2775a51f5fdec2145ea8d6fb3b7204397 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-ca38596aea1b7a4294dc03b12239e3048b55ed0713436ebbddfd925da8ac3bcf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-064a23f0 | sha256-f923fa055cc127a94929b5688fbbff36a6665833413bf7dda6e41e933c117de1 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-064a23f0 | sha256-f22ff7d4e564f23752c648f509127b96e0b801f3b353fc266fc6a712f35d5383 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-5dec73c1 | sha256-16e3ee1d0956e6a1d794a6b537cbf451e74409cc271bd941abcd7962386fb5f2 |
