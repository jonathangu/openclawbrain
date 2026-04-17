# Recorded Session Replay Proof Bundle

- trace id: `live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b093cf106a437e24ba93bbfbea56317e62afd65cc282953b847c0fec17c90f`
- fixture hash: `sha256-f186a663337b28243cdd6e62a9c63e0bf0678cf05202237e1d19a1f17b82f110`
- score hash: `sha256-df7e9d821dc1bc2224c4a9bfd80669ca5eed117ffeea22caee82de64fbf20cda`
- bundle hash: `sha256-309d84e6f6eaad9ce6c0c79809df6f926f4ee1b06b7074e1dba59aee84aea609`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea127012751163ce5c5c7b6a51409b045b05c15be13611d375e11b98fe528366 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-db4006d10a59c5757e9af4567ff5dfcafa1c871c388bd93afc46bf60d35b0cae |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1923ae980e5b8b263bf46604f99e65f53e660df535f15d7d4b74035d39c1121e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-bd546b94ec3a83f2be9ca8e6311ebe2676c1b5b2ca2e8e98888e3a77f8091ef4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-81c2eaa1 | sha256-85211094bcc4385557218520c4808fa95988e54a0e06e387fd84769d3bdb526e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-81c2eaa1 | sha256-7432fb8061341d06d1a5b15377d1019a75843212b22365432a68a0942efd7740 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d4326c48 | sha256-6cbd424d72f388655be50804f57b9ac100f0e427456523b51bb70cd9b442cf9f |
