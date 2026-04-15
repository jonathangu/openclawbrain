# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-529373cf8f7314054ee5a9938b5133a303e70a9153c03b373cae4ff852f394c7`
- fixture hash: `sha256-c16690eb3752325552dd8dd957f6a57c852c3d697d1ce7463c9556556d92ca19`
- score hash: `sha256-90d7b2225d849541634a0c782bf29007b8153ee5958d0db31d0e493c6e57b1ab`
- bundle hash: `sha256-3608a547ef2884d236ddaba2eff335bed2c1ff128c93cfd60728a68736a54bae`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9641da18215ca2d07fc313a19aa471e30d85d3a5754d470ceff969f5080d786d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fd96d039e30387038717d33860c27ca252933eccce19cc80595471c4d1427a73 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4900ae3a5421c90a61005e810f8c16c268fbb7c8524e29f324c75437304481dd |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2ea619beac7b8b92b827d800ba92bddc346c0bcfeb09888cbf735cc79b293674 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-00dbc058 | sha256-bf215fe44e5f3cf732f3d09490408df251d2d7084a836aa6b79696219a358b71 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-00dbc058 | sha256-d04a0ed225722de3b62c07bf3d4f9dfb4126e738fc411b16c9846710041f8bf4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-00dbc058 | sha256-bf215fe44e5f3cf732f3d09490408df251d2d7084a836aa6b79696219a358b71 |
