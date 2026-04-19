# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-74d7bfdc59dd3db31bb5515d27016afd37c92da2c7bc4eab4e2eb908b0aa9b0c`
- fixture hash: `sha256-cb31ecdb2e85be5c4d11a69e22103d461970f7e7d752e4a1e0598a4d80c4542a`
- score hash: `sha256-5aa5ba3083d04e36be08cd2719896221dd5a9cfad9c5f180b63dabb3ce9e6761`
- bundle hash: `sha256-c427b697a5590571b570df6777b40c9a8a70bfe65b2cdffb93586d22a330583f`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-629a5787a7df922c0732d245b071087f795fa4094d553ac2f295095a8256f812 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-5edc5e74047b6305849ed4488bb875f7abb009ca723628d36471af93847d3dbe |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-d05f60c076eb8467da17743c9217d1d22106a0cf887f8ca31d55fb57ffa5bacc |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-82bcf65934b085c45de0cd3a5a886e376c20f6d1a8b0345069cc54b458beb9bd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-5a27d519 | sha256-6b580e125439b45d0ec19c8cd899f9a3a2c990c9d3f1815ae3f13143177ead88 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-5a27d519 | sha256-f9c3bd61c273118f75e1345b90ea9e3fe3ebc4b5911c29318e4542d6853a354c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-5a27d519 | sha256-6b580e125439b45d0ec19c8cd899f9a3a2c990c9d3f1815ae3f13143177ead88 |
