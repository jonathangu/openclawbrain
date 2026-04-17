# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68b2fa5dad518df8f3866150e5eb5ae2df4d40d0d1730a7b39326babb425756a`
- fixture hash: `sha256-9c558cf390c2d5519271f6ba91a97c5aab0727de8cfbeaa1362c2e39d2a00c20`
- score hash: `sha256-0ab349ddf797511eb6a0d63ab791d9b733554212532444f3eb3f545407b83a4a`
- bundle hash: `sha256-6e4a4950bdfee694deed31142fc14311bec0a9def2852f378c029f31db505ad2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8709c6589a780862225e4afa90cbbf44ed4ef4f7b39772bdc54c0a9f8a33087 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-de122ea4ae108cbfff555917ebe2656cc6b58fa812bc31f2a938d6b16f36243f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4044215df5d16c38cbc0d8c45e65add799627f09e2be1660c8ade90592a5c084 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9730b7b423f375e82d00a60d93e28d3e497deb43e523b7012dba57e3aebc8822 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c4ecf03e | sha256-1673491deec3d47f9e6e3b76b22ca9f40806fb8e17ac06ccfb6391071bc512fd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c4ecf03e | sha256-41db7d6e629454ec51984b73e17ff83a92958bdd12575216834b29ec4dd0f348 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-31aac62b | sha256-33de207d1090f8d5b7d9e8fb613d810c12f9ad22697f81ce40ff01dd5f0063d9 |
