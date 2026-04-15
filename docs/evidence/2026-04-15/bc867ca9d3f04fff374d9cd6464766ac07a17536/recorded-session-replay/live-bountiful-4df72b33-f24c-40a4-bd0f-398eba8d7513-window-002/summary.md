# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-25e681dde9bf99a5066e3fd272c254e137908dd8248f9cd30c28377b5642eb80`
- fixture hash: `sha256-118dd0d43d47e09d3e0fb14557115fffb91ecc9b2c9362bf193950d5af577035`
- score hash: `sha256-33852c7f42b949506cb5817b5fd29e1da3ab2d800597740146fdf183d4fdcbec`
- bundle hash: `sha256-267fed6ac36a2d8ffccbe95c8b27dd3fe9b18f6576c53109b4da3bce9475c88a`

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
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-053f07f407b9f0886975eb3e4d95aa7c39bed9e8cf96e6716ec7a7f71273ccd2 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d5e633df7a122affea216721d3b4d0d7f37206f5e27e27466afa5289ff44864 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-d801080f98bec60553d83c2e28f95be0dd705e8dd53e129df9c335818895d6dd |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-efd80b866ceab6c1b8c5914c46b5e44318d277d4e5b841380957bdacac58225b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-f5730103 | sha256-1be6c90142bb0b4da2d224189b8442032eb15ecfc1c304b8d1977b1a652bfbae |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-f5730103 | sha256-1be6c90142bb0b4da2d224189b8442032eb15ecfc1c304b8d1977b1a652bfbae |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-f5730103 | sha256-1be6c90142bb0b4da2d224189b8442032eb15ecfc1c304b8d1977b1a652bfbae |
