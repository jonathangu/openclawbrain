# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a6c28313900614132b45e0ea565c05af51ca312d324fdd7ea457443cc9732010`
- fixture hash: `sha256-33c60ab5b3d1fed7da251c9623ad91d7f552d5ae6e358c91f60c002d0d9e41c0`
- score hash: `sha256-c98cbac39644354f870bdbf9ddfe84915ef97bef743dc7a9e78a3d8693e0d7b6`
- bundle hash: `sha256-a149f3833f1eb2dc9317479b7160615f36ffdb80901d49c6707540a6e2a3b799`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-43901bec518b9820aa1206ca8d4af0fd884bfda276fb97e746b980a65cc6c82b |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-63ebf7018c2da26b4eb769c96f958c0023525e6c8ac10639941718bbd81321b7 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-bbd50dcd19ddc45e5f8a1491c8ed5712d9d0f5ea8258cbe7aadcc1f9f6190ece |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e16979f4adf825bd692f3e5c29bd35333c914f76815dc3952538267b24e724ba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-b60ad0ec | sha256-83fe732e015b9f405940ed09e65d8c4eb1fb9809798d64e4324e228fd1627381 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-b60ad0ec | sha256-7087e37f86b989a030dfda8bd84f036f792d7d4cc37e3d2a2fe724977e87bc8a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b60ad0ec | sha256-83fe732e015b9f405940ed09e65d8c4eb1fb9809798d64e4324e228fd1627381 |
