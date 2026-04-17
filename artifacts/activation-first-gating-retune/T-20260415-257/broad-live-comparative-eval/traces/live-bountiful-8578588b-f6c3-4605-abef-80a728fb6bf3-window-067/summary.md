# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c03db6636813d15ab314e1036640845c998fa263e7aac887b43bd18e611eb255`
- fixture hash: `sha256-a5250b921f3f7515a2c7a2c53cad821f0baf9317cf0101574260207ef30568c4`
- score hash: `sha256-39f491fc9842c9469532d2d40c3bb8518553f69f376119480ed4fb8634ec44f1`
- bundle hash: `sha256-e4f07b609aba4d12a6ff5106d7e910b9757974cec8f726bfa813a48619de92d3`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fda1d2fc29e8cba170ae68633dbee693f7e6102e2f256c7ffe0a2134e709581a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d9e42141d43b2db965178eeb1d5ec814216c101553b005234201e15849b79be6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a18fb58f18663f2b42a1ebacfcdbd63dbb4b40186e3e61f8dc69e8f12262e6c8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2cec1113c06da70c4bfcd71a25529a54bf18dd16fd103871f1b456ea79fd0b48 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-94b67fdc | sha256-9faa171d330a2f1c53607c4dbbbc5447f8b7db6021d2137935e62e3462648ec0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-94b67fdc | sha256-a2ad5624179107cdf944686f727e82a42fa8759c93e33cec14a5d2db3f55313a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-4c921c2d | sha256-00659b7b16f2c5e819a544d042245b3286036fa5ddb105f22490323300be5f6e |
