# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cfee0e7c7d7a332b7f5a4f24cbf19b04a7b31698d12d254a92d62753684d371`
- fixture hash: `sha256-539cc58588045f4d44638a17795295875c8ed45ffa9d4d266b2c19df9a95dd7f`
- score hash: `sha256-2cc02f347f64d7fc7a40581094edb9599c523ef91210decca3ed50e903e67317`
- bundle hash: `sha256-04738c29cc13183a8b428fd75072fb6a89094a3f26a0e29389cd9b87c4780ff7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c32ce668e6813134ccd828363a96ac1b89f56519737480b00962b4a14175506 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b92172cb4ef9c2acc845f6a9184388b0e3c6991d9c0e1119bc6c2df3af34c2a6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-351705378f0924aa034d2e1732de96326160feb8c4191950925a50b738ceddc0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8f828153322209081dfcc738e78a174beda722ee83c92d6f5611a2014e1db215 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-00aabb02 | sha256-8ae1b9ef2aa8be10c89b6f658898fab647363e64d90911ad03305e7f659a2487 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-00aabb02 | sha256-c1c4e9744218b14e556b10ccd31dac070f9b7ce31b73252b6ead487d22a17642 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a9f566dd | sha256-d6ee545d1aa5278899c3b13f977084b0059e847688903e97988e30901247208c |
