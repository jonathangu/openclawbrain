# Recorded Session Replay Proof Bundle

- trace id: `live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6e4f37e5b65aeaf586e4d2a4e71ad3cdd999df7cfb455f741dd0b11a1a3ee8f9`
- fixture hash: `sha256-8f110e8a4894d421efbb2427ce24dd5ba84d98a2490639e91780761dd48a619e`
- score hash: `sha256-4bd82c31616b8e9eb7aaeb0355ac54466cbf9d5f0c5f5935c2cc9c88c13053f9`
- bundle hash: `sha256-a0b60bd866f5128f3c342b31288adeeab3824bdd5128e795beab9f6a611d4bed`

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-8e1f415f619139090b58659b2b57f0c38993ea04d1f8d555577fb21ebc04ae32 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-df1c4ac0e2a282c20bec00fde3a0dfa0ebb871558d009fc4b74cc62cad6b2a7a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c6ef66c4ed22902d6d1be3e085cf2001bef33beb1b7c0f3fd7280e3f6ebdbb2f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-507041c0 | sha256-bd157876ec022c81119ab58f559f594d7732ce6cc8d10c2e15d02141a3320c67 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-507041c0 | sha256-0d4c7b0f3298c0b76c98a27885fcc4d03ee837e808668a759e317155749c437a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f938d2df | sha256-dec439adfedb975ea576d045431f838f528359c3f46744714cffbfc33a6009bd |
