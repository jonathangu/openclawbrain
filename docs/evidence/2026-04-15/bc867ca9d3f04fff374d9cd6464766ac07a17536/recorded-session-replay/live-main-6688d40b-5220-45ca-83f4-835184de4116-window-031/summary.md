# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98ce4509785da1d3e9688496a53303f79675442a91eaedda79bdab30b5e6b8cc`
- fixture hash: `sha256-ab905612bd3cc43deb68d413a855b981990f021bcff6e0685761c3af602b59e1`
- score hash: `sha256-3995ba16c8f08461496f6741d1e87f93d8b7def4b93c3823cc652fc279a07521`
- bundle hash: `sha256-63306b97ef0c13ac55f040228daf5e42e7dd6d331d9393bfbf01d67c4a2977bd`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16539ac70abd2ef9678c6c7835bb8d35322c600e9de7b2b4d16217df707851eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-176753d57fb867d012c51b875564511ffcea045a9c86af7ed6356abb1b2bf443 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f57c43f272f36371bd431b85cc1fe2b4fc7ed3793c93aea158eb401b4c52491b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7e312979eaef56a5b5ebf98a5550e78507ce6c98df3dc5cac7574a2d77f3d0a9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e1743fbf | sha256-ef2aa7d9866ffd8477442451936955fa3ed81478911c8b79417edd7a0e3de613 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e1743fbf | sha256-672dd45f1a401ea9dfcdc172eaad5587f842f225803cc3ebad59e03aa800ed7a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e1743fbf | sha256-ef2aa7d9866ffd8477442451936955fa3ed81478911c8b79417edd7a0e3de613 |
