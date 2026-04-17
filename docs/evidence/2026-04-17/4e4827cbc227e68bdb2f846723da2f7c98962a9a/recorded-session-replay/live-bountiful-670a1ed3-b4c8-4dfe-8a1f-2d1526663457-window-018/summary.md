# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8731aca670fb1adc2a11de661b208e90de02229e43a59b819be0c26634995543`
- fixture hash: `sha256-b091c6d75f126cd4fa41e0e62e2c1bde2a5cadf897b977dd808714e16a9eb7f9`
- score hash: `sha256-3f2dccd8c5b5c0092fad23c3651008e8c58db8d2092cbdfe36220691665634c7`
- bundle hash: `sha256-480e739c0229580e092da7a60ee0c7b5d5d2ba08ca9a2ee5a49501f81b043d20`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4dd25e120884595a4500dd8027a1e5e49f93c256e2e2739aa127521c9309576c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b007718e2f44eb81230fb4b22742c0267082ea2fe87debd464649c7e1a7e031a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f75399c5e95642f32ea173ba8d67b971dfa3878ff21ab0b930b59c8a73050b92 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-cd8f5608d31ad8574c37e2f6a7546dcd1edff3b132936a5013ccf497bc1b68bb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7dc699e2 | sha256-1e39acdf8c6054259bf5a9e9ded6d9563425c80c175d3766654c84fbe5ec5a13 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7dc699e2 | sha256-de8e9bddbd5610fd800c23a87194899a9721187f6ff764585a65ad0b486c1a1e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c5ecadf5 | sha256-815bf7e152bf9f1809aea9f535d5cda11d599977f2a5463d225c16d71abe34b4 |
