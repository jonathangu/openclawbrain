# Recorded Session Replay Proof Bundle

- trace id: `live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6e4f37e5b65aeaf586e4d2a4e71ad3cdd999df7cfb455f741dd0b11a1a3ee8f9`
- fixture hash: `sha256-8f110e8a4894d421efbb2427ce24dd5ba84d98a2490639e91780761dd48a619e`
- score hash: `sha256-087eb386f85b2063224a278ba58f334aba9ae9f295ea456d7926236626bc0199`
- bundle hash: `sha256-b32c8b0e3f5a2683ff3dd1f23193561a66fa2d859c68e29027ad267ed1d55abe`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb1995b9d3380942336af495a709abdc9277059dafdab742d11d02fd9c054a90 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-490e9e5653105c2da3d3f370f260943c3c86b7edcb1ab6df55a6c79c3c40b777 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ac0daecd7a2e505d4ddf1904f43c75a29b3cb3c60f98e4f039a238f97cf4c4bb |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-aecb69c856eab7f75fcc689caa32d41cfe0cf36b6b43e60628b6435c2930cbb7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d432d08a | sha256-921682d3923a4958b8e1089baff76495e0dc0d8f77b934a467d30b0c469d8771 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d432d08a | sha256-1634cdeb3518b625d062891b7bb700774726063e8524c2956e569c8c347052d6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7cfb61a9 | sha256-30b13896006896e6fb5f4e88a51ccadab92279b975b2929452c6bd905d9ab414 |
