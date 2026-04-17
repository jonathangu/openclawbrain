# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-19cd6a701f3afe5404567d59955346d7cfc26c77deb7b29e61fccacc22d3bbfa`
- fixture hash: `sha256-4dda7357e5652f879faf39fc4f606d23e6674326c96ea6b533ba27ecfc72cf16`
- score hash: `sha256-6f4e3e56d166fd417cc78714ce49f3971ee8d3a4f06f6ca931fb49ce416d92e3`
- bundle hash: `sha256-50bd08632bb24b2dd2c46fa7eada63e8d8e7e39cfab48c7d15824ce71fc1af21`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-934729bde748377658ef5251e3c9784137a24d5cc133cff448c2ec475fa6a4b7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-768e7543c50b8f01e4b5c45eb15992815a393dff32a8899bdc41f317662c2035 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2971a3ffc81c30b7bd1664c247604d51b201a520e5db3aeba1ceb71ce40abea8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-38eaff93f46011ef5d7fda882fde9e9b1cde8429b886c16bce1343fac20c87d0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a356ec8e | sha256-786c987c0d08e7d4e7d4a3ad4c03885b3af339e2e9f93fe913a4a19072604adc |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a356ec8e | sha256-34877e6b50a123dadd830e6b752e98f03fcae294eca5870293438d1e55a4496b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-86d249ed | sha256-b0e0b9637eb2dd07429d9661a474ecd873ad422539a2ba737c66ace905cf2664 |
