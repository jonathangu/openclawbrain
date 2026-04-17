# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c32e2ebb8b7b9b5fb8d50fc85f34ad187e87c05bcad5baa201a1c538d7a405ab`
- fixture hash: `sha256-e23c11c000ae3f195bff5e2ea98696c33b399e18fdc28541e3c50f4b667d3e58`
- score hash: `sha256-706180bf72dfc6cfbe7aa5d00d67a4949a597020c71080afc967359f9cbb4e0d`
- bundle hash: `sha256-c7bfd8718cf0ea1d97ea36705ef6bbfee2e7a914fa077204c135afe919f004c3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3b40ea9a22b9a22398f998c23b04d742ab7923c5900fe44bcea6dd68bb464780 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-916b6f78dc8988fd9754da8fc3c39ada00f4ef77a9d7b9766ec278e7b653789f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cc45e92b12e6c7dae24b125d8609b0e87c71b0914fa7f6873b7275af34b90095 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8af2ab519f760e11982ce5d88cdff360db5fd4ffc08d7e64951b495bfe1a6187 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6972456b | sha256-67088fc8db206a0e312a184f853e598dacaf7fd6159298eadfd6d9a3ad4a8f7f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6972456b | sha256-31063718162923d69dfd8fffe7680d1ebc0ef9d199fad0a48830f8fa1d3040e5 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3893ef26 | sha256-e242b0e85c24a3c32e13fecb8788ef3df0b84579f30111bb9c90ad86fe3bba4f |
