# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-500bb42a51fe35739e28b1f6be3d9fe7ff92c6a8eeb2f053f3018ae2eba88584`
- fixture hash: `sha256-f69dca5c27c722f582ac3debb2e25adae4c35c5bd6a4749aa476e37eee07c7bc`
- score hash: `sha256-fca5d49daff250888dd46e626111061ffd21917f28e6eeefff4dac0b99f607a3`
- bundle hash: `sha256-4086aa51de447a227dc5e82e7e2bdf9873b3acd36b7dbeaab663b19baea810ba`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a93c94aa4cb26ac67e3ba4bdee5fc22bb0276c3da7ff11089c43e42405c272c |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-6dc70b6111afd4458829c71e76ec2b7c3c3779d648974d75c67227008b18f4c9 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c674d67bd75ed5a343bb4f1c07962bf45dc348c98c176f06f33de0fbc1d3de63 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-9c2c0219349da85f076e380964be27cccdd72cdfa711cbb536459b0b7cde3256 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-358680d0 | sha256-346ad3aaa6a7eb9ba0e12eb475b47d661d23d8a8d4978de40ab5fa48bcb4cda2 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-358680d0 | sha256-6f1bf3285270dda67df519742a1dde126e3911e627663f6faaa179f946d4774e |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-358680d0 | sha256-346ad3aaa6a7eb9ba0e12eb475b47d661d23d8a8d4978de40ab5fa48bcb4cda2 |
