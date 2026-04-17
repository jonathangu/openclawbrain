# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dfa81f4d2b00217c5c5c520178573740e8780c6997e7fbd463fe714331cc7869`
- fixture hash: `sha256-ed00dcfbed6598ace12042db40479b3199c9a2955a7a673a786b8d8fa048ed17`
- score hash: `sha256-226258948da9331136766819f7f05cd7bd133d456cb0bf1f12c291c11dced63f`
- bundle hash: `sha256-0e9575291637269514b4e13575f570ca69a1b42f0d0968e2098b6789d68dfdf9`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2d5028651743004fc65c4abb7a18a3ce781f93f13bd67703dbd698c51e61ae2 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5cf91e0d5a70ef7cec8d3da0d79e013027bb4fbb6fc8697980a39f348681a12e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e981bc1c4a392f1504d6a5256e1355663473eeb04a0a14050fcc273c62aaac5f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9bb4c1261420bea731e74f761e1c1d18644fd03cd2757a98522af636aba91e18 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c9393112 | sha256-8d93e3a114b607c52d143836c8661db3ee93a134ec1545cc5ce5928e70fb29fa |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c9393112 | sha256-8d93e3a114b607c52d143836c8661db3ee93a134ec1545cc5ce5928e70fb29fa |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9b708231 | sha256-62bb2b7d8c5393c576ac3f278be70a84497521e3c18da737cf0a25864934cfe8 |
