# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9af4d3068fe0abcd8b0d002d37c1f3cf1f47d195e7f9302f6c99d8ac1c1ba8d0`
- fixture hash: `sha256-b1358b7f23f888234738e0f7490e996569d9c09e6a59451858fe36290e0374c4`
- score hash: `sha256-f339cd3a2a3465b5c1950697c5c929890a87b7487fad0d658978270646ee9d65`
- bundle hash: `sha256-1b551b9792ce404edd0523a70182604e990cacaea1659bccaba82d501b65a3a2`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-38fcd5e204f9ce44394f224032307e552dcb6c83cc2ff3a9c8d07b3df48aab19 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-18c6f29e661c289f56b3fbde4e3f4c195e1f1c38e5d809a19fb4247169517e8c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6141fb89d252f82c6214eea50b53a477ba9934903eff22e3edac7bac344fb866 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-5cd353f3593436429838449c081b92a65987dbd0eb417a8f56b166c2f46881f9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c641aea5 | sha256-cb74b91e19cf466fdcda8509ac1df4adb4f2d5ca090083e7d5f3f232b89d30fc |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c641aea5 | sha256-74d1fff24f4cd57edbe022ba15ccb1cdec20410954512cb04f4217bd5250b73b |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-2fb874ae | sha256-da0817d4280cc48ab5127eff6f51f903573de44e22a07f90b033e3aa316e4824 |
