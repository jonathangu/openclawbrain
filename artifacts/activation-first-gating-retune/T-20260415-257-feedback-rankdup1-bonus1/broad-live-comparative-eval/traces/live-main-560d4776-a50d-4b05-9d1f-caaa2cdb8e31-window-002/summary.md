# Recorded Session Replay Proof Bundle

- trace id: `live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-035806e6f9bcb3753f58456b70d56dc8a01f4abf60114aeaf359384806f6c24b`
- fixture hash: `sha256-74bbbcc2ba3e23b87dadc56cde438b46daa30c3743245ccd0b40d24de1249370`
- score hash: `sha256-939213bf8372a21ccd2455ff781cdcf7e4e063349cbcbe5a8f2f83b194bb13c1`
- bundle hash: `sha256-72922de513733230bd3c5af90efaadd4aa890d497b23c3bcdb15b98401206b60`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 1 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54ce8361598e1b4080ba115badc91e906441ece2076bc91bc1f9f28df2706034 |
| vector_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-ef4c909c864090ecfe3c2bf41922880700d21e4927cb4a278fa068fdb205d816 |
| graph_prior_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-30c3e572c290d364505558942c5d931da2e813e2dbdad682c8b5291be94dda6b |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-08352c13101852c8594931d603a12fc6c3a66ff69a6d1b42c7823d396f45c51b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-f2341fee | sha256-82222e5f07ed21f31efe76830c460ce6704eaa0e369f9af63d051fc016fad4c1 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-f2341fee | sha256-227faaa0bef58ed1e3c6dd75ccfdf8d16b41436bb457231fafa03e5f04b55e19 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-f2341fee | sha256-82222e5f07ed21f31efe76830c460ce6704eaa0e369f9af63d051fc016fad4c1 |
