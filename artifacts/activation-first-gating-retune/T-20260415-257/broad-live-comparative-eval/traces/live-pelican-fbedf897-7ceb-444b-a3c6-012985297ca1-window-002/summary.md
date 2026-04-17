# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4e6bcca0889e112786ccd30d6dc08d693afeab2955ee3e21db9f09dfe3094e0f`
- fixture hash: `sha256-14ff100a8ccae36fc1c57494dcba2b6e1338cfd708e5c890121212b4f7b539d1`
- score hash: `sha256-b17c07fbd3dfa8663b9e6d3faeee0c81b35c4b6f2363b501ad2f153bd122cb7c`
- bundle hash: `sha256-8f44b1987d8eb0ca6a0a7d5cf9d6673fcc3f7f4976906054281180d919492c93`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86a96bb1f89c4f625498603269cc86fe2157c50e9372e11582b94c39873a6510 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ea9842cb31c2dffd0f6420bc822052c878aac25c2d67c15ace4766d2edd36432 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-786e3f1063808cca40d6b21e968d88a7ca9b1206c94273d8c5fbf00a31ccf790 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d6a904fa1bb9f166db9392f1d7e67b9c6e0d2aa09ae4041897f563d354cf7b08 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-81c2d9e7 | sha256-f4b4be444c6599524e96d9717bacbdcaa51f889fb01895828f7ddc40834f7c6f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-81c2d9e7 | sha256-78fe56e0082d7f4b4d451ab977f64d2c61a6101fee82efb3bebc6b0d9ea72254 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0def2f5c | sha256-53b50d49302a4a695b1befb7e0cc1808e65e92a24140e6d072e13ca6240c9c7f |
