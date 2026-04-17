# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7c9bbe1bc32703bdc0ba57cd7c2e5ba0147d232db874dca18f8d1c93a644936d`
- fixture hash: `sha256-1632c273e7fcb25c5de9fdb5adf5c07fcc4c43677737f0e63cd97217f3d6d9e5`
- score hash: `sha256-15b8f8e5f3b1240f783b7962ce6d2f1811f3c476b7ec116e8710a1d60e2fd72c`
- bundle hash: `sha256-1ab0d6ca86a3a95b2a8bba563c9847671c794070bf4ff71027dd21b28dc15801`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-80e9af933d3a18c2836442131236b812d1fdf8db3bb96c2fc77c951fce5a2ed4 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-28d7d6c71e70f78af3deff5de45f6cd0f4b39a1e436429d2fe46ff29d6a7ef40 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-20c105390c129538668ec37f1d1e823eaa9748cc42f793df4eb0030f78e26d34 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-aebd85427919d757dd410c0b4f73d381fe5434cfa9c5da1a90472b1c074e752d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bde214f4 | sha256-aad4e72860a4b0588b784f7f24e88d61fd5a986153b0773aac84b71a44615d2d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bde214f4 | sha256-b874c518d8bf7e5680c656cf97475954eccc07615ec3c0a3c122a5dc4460a6f8 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-73938243 | sha256-d20f51bb552e4f066be60534e59e8957a08bb0b899f32e4642ed33e8bfd1d0d1 |
