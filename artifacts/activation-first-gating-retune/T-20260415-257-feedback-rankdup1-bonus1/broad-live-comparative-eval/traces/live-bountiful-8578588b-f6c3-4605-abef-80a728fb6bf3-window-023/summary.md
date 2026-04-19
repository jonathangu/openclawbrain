# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ca0eaf6119fd5eca026a2af3b944ea416d6b1efe4eb9fefea42b7c6ff57e6bc9`
- fixture hash: `sha256-3b94f7345ae9ef307ded464a5d75cd2634838ce66abc47afb09a10bd7f7fb2ad`
- score hash: `sha256-fdbebb10bf36c2f6779b1613f5a4921a5d3eb77cfc11a3acce566634ca57bbfa`
- bundle hash: `sha256-30b26855205c23d456bf0be3e1a9db02d166d990b3e55be6feee7894794667de`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9867396f82c81199e6f66038091f8c07ee4ac9568ec27c50c752857517ba3f4e |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-ae6df6bef84ad61cc5ff3cb5f610417864ee7f96a87200271775c107ad817913 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-94308f16bcf33efdae92957a477d707999aa5bdf8f829bca19eae037c9d40da1 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-6b3770b392d738d2bf1fd4be3de64595720a51ecbf0a8f6be817e9b89f29b932 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-ab790447 | sha256-da08e3421a5b85ea570a51071c55b7c05ec30d6c140c5ae2c94052354ac99191 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-ab790447 | sha256-8aa80324ccd4b36c8c3002d1fbc4bb92a7edee2b6570790981ae4c0e854c6eba |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-ab790447 | sha256-da08e3421a5b85ea570a51071c55b7c05ec30d6c140c5ae2c94052354ac99191 |
