# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fc1409b104d617856751474f01593056b66d1b2ca492e8f5dd879839efd10f66`
- fixture hash: `sha256-8310747322d42de0fb2d06597a429aa5eb75a2026f88cf3e458dadef80911084`
- score hash: `sha256-ea311c8a8448d1b791693532c23a34a096fe974ffca04c8c36eb848fb30af740`
- bundle hash: `sha256-90ba20d68ee26f1f4d961fb91e05b563d85d37c1859f51f5dc42c105d40a340c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0fcb97b42b2b441ec8190e1bb06fb82b8bdd1457d8fd6d8d105b2684066c5870 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-05ff33dca0ad088c31ff236f8d83e4c5ba4277c47330deea6a21d4f53e5e88e4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-51bfacd17586e8ed244814dbbbb39a2f5ae27a8b4539f41162990eaa12c8634d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c7da321f78ba8440b0d677741225fda52ef6ea06b6f0c1aa8cdc69132c55824a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-55518928 | sha256-8b8ed197359c2194bc4c6c4a065eb1e22eaa6ade97f6e0e7c5e603451b72a489 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-55518928 | sha256-8a9c62bc1a5befd892a432fe193f6741c2830dfa0549e4f9075bf0f51656f5d9 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f8526939 | sha256-2f4c8202f0fae84dbec82efcdfcb0a17fcee5c04b432dfd64d740ac74286f118 |
