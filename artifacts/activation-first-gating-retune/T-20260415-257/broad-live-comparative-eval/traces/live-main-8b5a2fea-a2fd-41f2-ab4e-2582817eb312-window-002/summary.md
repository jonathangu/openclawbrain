# Recorded Session Replay Proof Bundle

- trace id: `live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e0e56ffd1c26d20085e7a9eb3248f58dfab8c43d92d6bc35e804da203ef4f7d9`
- fixture hash: `sha256-e4b8d39277cb985d3e9ee559f9e373775182720bfc10b6d9350141f9c5016460`
- score hash: `sha256-492d374b42d9c1e4f44d31f9427b73409235a8a1653421b013f116876b050a48`
- bundle hash: `sha256-2fb08c4c787408c31b7a1e68bbcb265c231a9393fe68239bd6b3ae4ea39bfc30`

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
| vector_only | 1 | 1 | 0.666667 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 1 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0bdf6c0bfdc77dfb35df2ddd80b080b8e6bbd2f8f1020fedbea4770e769e1c72 |
| vector_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-cea6247ae645fe61d7dbc2bd151a65f45520ae4910d5fa9dc2d14b4a9aa47202 |
| graph_prior_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-88d1880f1cf98602452a8b3549d1d35b935791da0cdb5ca89ce4da18a80d7267 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-f8638c44c117c475842fe83bd3ff4eff6f8e87341f11e317a2b3414287bbf0ca |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-66fd9085 | sha256-c0a12f44ad4613a8665ab68f1a0cba471d49a3632b42f9711c0b18f83d4aac28 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-66fd9085 | sha256-258487be9bf408944917a7597d63ffb7b2164d8e79823a1a4617cacd6c908b0b |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-66fd9085 | sha256-c0a12f44ad4613a8665ab68f1a0cba471d49a3632b42f9711c0b18f83d4aac28 |
