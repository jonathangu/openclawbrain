# Recorded Session Replay Proof Bundle

- trace id: `live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-906cc40ece3b0fb4a531e168af49c80ff11ea62f07a752db4c8924f98d189aca`
- fixture hash: `sha256-565a616fadde1db10f7ad35acdc4ddc02cf8260e0bdbb94b6efea52c6bd1c593`
- score hash: `sha256-601e0831c419898937dfc0f6cae590ecd4d9b2052c157f44205f4e5a2a753f1d`
- bundle hash: `sha256-8cabd57f929f95cbb63be4288a27cbf715e2102c97e8d2b9be1c05c947b972c7`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-faae0cd9c1a72b5cddb9e2597356cbd0162b3076b86d83b9efa7f531bd948257 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-36455023d276cdf176047d877ed9de27489621c35d57c6cc6993ee213eab798b |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-3b1919c91e870b467d45f337f8c84722500e15e5bbd523c455392368dd3e4714 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f559af85b41eb407b7b13d561f519316d2497299f1d0988a108680f060e1e535 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-d2bcebcc | sha256-587123fced8e17355cfe2a76ebe0b85ea27874f426a2de37f81ef614cf94ab91 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-d2bcebcc | sha256-587123fced8e17355cfe2a76ebe0b85ea27874f426a2de37f81ef614cf94ab91 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-d2bcebcc | sha256-9f26495b9b67de5178140a0e556ea3543df6804e8156f89b758d03b7fef2afee |
