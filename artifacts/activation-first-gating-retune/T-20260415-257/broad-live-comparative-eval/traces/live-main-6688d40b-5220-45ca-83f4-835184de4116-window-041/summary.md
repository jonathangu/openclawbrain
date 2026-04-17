# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cd17705850f5fd87f770e4757922f483be90c3dcc5bfff44d696c49e62560cb7`
- fixture hash: `sha256-743937076adce554085fa9dd3236567f573df76180477a11d06a07f43c4044bc`
- score hash: `sha256-309bffe342ad23259ef1be8a05b5b4dc6e333343ec75ee9e0f71e4b4e3c63a17`
- bundle hash: `sha256-8a568e809025c9f0d4370d665a55da51c3a786891c1de0f4cee56d9f14e7511e`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-38ffbd4329a21a765f40f1a44ad7d1cc0603504c91e4e697e7b573151d0b2478 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dc6a38f88b8128cbb0ffb913642d91b05d1094bba6f84e2d1e9198542603bd80 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-064dacdb821b89ef1c8a9f3b0c001fa978abb15b538939c62d09161db7f6ca49 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d62a00652a429290ba6f2e3b76c689b90084bccbbfff074a347e83ea58074b93 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a9b94549 | sha256-db23c4f7f0fa52d428182d00c98fda2bfc966ebeb9a6ce4e9cec1756fb5feab3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a9b94549 | sha256-01ad9f3a48ad70688819e02df6430aeced725ca53b71a00455e9cf2fe8d62928 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e96dd222 | sha256-ee81d75bd622e94d95abf3978186d352a5e091e0734a20254324ad6fcdede6a7 |
