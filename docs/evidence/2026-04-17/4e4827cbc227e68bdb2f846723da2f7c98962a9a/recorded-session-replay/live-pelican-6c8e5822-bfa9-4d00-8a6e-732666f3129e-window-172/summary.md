# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c110d52fa9d814d5415fbbb31a6466d9c241d27a71e256584d4f8da38b7870b3`
- fixture hash: `sha256-7aea2d0a1eb139bffa0a7ec4a62af3e2b3a4882d28c7223058abf7f69edb1954`
- score hash: `sha256-ff895f99e6c1f438c63c2f6de972de4f9ced4e187b75fcab99c8bf2000bc5db9`
- bundle hash: `sha256-a15362260abab05a6a60b58e4acde09781a76d54e322395cf46dc9c8e2f93cad`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3c530b06d54c1f577e737bdbbc2ed643a0051ad5929f4a0256034473b3d96cb |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-63e26e1a96f02573ac1e5c63873effce407a397ec002333817e9da18e8cd582b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-77a1878544616248940d9bf648ca157e93234fc14f736fee2642d397de0a3ae9 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-4cbecfa5b7fd2a847b94febc9a8d0548c8864423fe44050634ff232ea4e4bd93 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e3011481 | sha256-e0e916df29da3ae8a3e6fa096a2eece081ee67ac0c71ce584d460290d55e80d3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e3011481 | sha256-2e1c303359e1d2aa3de43daab0a9e81fff1a3760f483876e695ff3805b25febd |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-0d58b3fa | sha256-3e0047408e6a4c8b4442857a640e953d75f3624c60ea11f9bcd46bbc482ffc14 |
