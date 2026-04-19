# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ed2f9f31e28bc4c542ba13fc0a4ccba3e6b6e5db3982235d09f16d62242d7c5e`
- fixture hash: `sha256-c571aef0c0ac7b60f97a81ecefc88f95d1024f6a761836a503482febdda1b1eb`
- score hash: `sha256-22b15c0d961a1120c1cbefa03ff7c9f67e3c2a49af759c29d1846099e0d696f3`
- bundle hash: `sha256-16abb756b7762ee8b2a9ad4a62fa9c17655d4407059693ec2a99ed7bd0892231`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 1 | 1 |
| graph_prior_only | 1 | 1 | 1 | 1 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4a371a11e3f0400310e154f8ea3c13a532ee5c397c446eff3697fe01cbdc026c |
| vector_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-7733842bd79fc58098ec52bfa9de1381ff1b028d3a825497a5679a6e7ba8f531 |
| graph_prior_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-a0aac5a42ddd38837ec64d86751c5a3946868484d0574b1256d3624f3c895c0e |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-53fefa4b7cc3bd298f3f93f9f1d625bfe098f5090d0a73c338593163a86e940e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-a4fcfead | sha256-a14d01036889f5cfbfc76f5b8d7d3a4cfbebf3a67020113e0afcfd331d9da7eb |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-a4fcfead | sha256-d2d7966216aa2ed03f0b7a7e9011750d7bef2ae7fb1c72df7c92e89c7a88d3b7 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-a4fcfead | sha256-a14d01036889f5cfbfc76f5b8d7d3a4cfbebf3a67020113e0afcfd331d9da7eb |
