# Recorded Session Replay Proof Bundle

- trace id: `live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6e4f37e5b65aeaf586e4d2a4e71ad3cdd999df7cfb455f741dd0b11a1a3ee8f9`
- fixture hash: `sha256-8f110e8a4894d421efbb2427ce24dd5ba84d98a2490639e91780761dd48a619e`
- score hash: `sha256-84af5f89eb3067b7d765abf4c68db91d02d3d33f5f6ee54d9032f4c2d58e5b76`
- bundle hash: `sha256-fe80d29329763cd94bc4f59f7d50c078d6f626ecaf0f35176b04cc99b18bedb1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-c7873f9af16bd9e3cec3c509bcc6ae0aa67cf2ea1d5513128f774bf462aba3a2 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-77cd4071b4ceacf137781d686b88c7d080058ec6196bbdec2ce4bd9a9b27de09 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-88bfbff36448a78dd1d8b5e433f5128a6f910423d33fc68c88f4948b41c25ec0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-1caa6726 | sha256-fb3193dd1f38314214f4142875282aedd0afa2370b4d840fd6cbf33b6461cec5 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-1caa6726 | sha256-67874db89b6c296b378c6e7ddc348bb92ccfbb3c0ddd4d8aa342c7cde0305036 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-1caa6726 | sha256-6d5502f10f85ef723731c0387509d1a7eb7f9b43fe6f0b93c7c611af91ade834 |
