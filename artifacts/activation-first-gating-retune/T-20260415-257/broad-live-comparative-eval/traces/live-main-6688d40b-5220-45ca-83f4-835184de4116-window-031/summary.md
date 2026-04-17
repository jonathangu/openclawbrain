# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98ce4509785da1d3e9688496a53303f79675442a91eaedda79bdab30b5e6b8cc`
- fixture hash: `sha256-ab905612bd3cc43deb68d413a855b981990f021bcff6e0685761c3af602b59e1`
- score hash: `sha256-95e91557ef5c17f9db39479ccaaad3594593b712b0fb474b5efcff7b1e5afdb5`
- bundle hash: `sha256-8369c8398e1eed0e568fe8417589d53faf2268951775f1edaa2becd6e0b7da95`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16539ac70abd2ef9678c6c7835bb8d35322c600e9de7b2b4d16217df707851eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e1034bfa15bee06be7800b2029f910648f36ddeb2e8af28ac92a152438fb32b5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-28a04ab01434a19e11f17171c85ae8a60b161c4d7168cd75d3b85dcc411859b2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6449c73cde7c33a8da749e2c9f4d7d5f481fcf8dc50dc24c9e8566f6c3190a40 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5214551e | sha256-0779ea54e677d2676c35a948d55d851c2b6c15430d0af5f438e6d6d43b31907f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5214551e | sha256-999f631bf75f6f1385c212116d6a12fce70fa5e1d9d379b4d4993aed2a1be161 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-c0c5d2db | sha256-f8e668d51c6059979f5dc01eb2f3e15f64407fd5326e33a2bb985ea94d88f221 |
