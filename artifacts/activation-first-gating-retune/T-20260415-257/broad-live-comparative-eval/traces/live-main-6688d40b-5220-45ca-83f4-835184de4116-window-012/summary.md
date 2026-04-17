# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3f29e706c8408e5460da5ae181547f400604bd45efe4b812bde36a617f82f5`
- fixture hash: `sha256-3bf32dcbf845b428f375103144110fdafde5982202bc1871fff67136d9720e81`
- score hash: `sha256-0b0cac34c5cc8a38d35fd7262b4410088120f48158612415bad1052d9cb463c7`
- bundle hash: `sha256-4c2c37f113f34c2a68015fee94dac2eafbd718e9bc2e145ecf64c2dbf78d6bb2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-865ec1deeaa418471d6bb216a38e6bca377292e05c38cf14fb63e270894197b5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-823a2205a487efdeb75919aeaafe2383af7c06e29b725810e7de04976270017a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-74f13f9717e75bcf887edebe6ed06a42361d239a698501b8ca13b52dc20b0c22 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-81f660440e9f0c7c6448fd5a8f0ee1d8a688aae25164d31c50ed47eb021ec497 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4ff25406 | sha256-f5cbefa40c367feeed653aedb780f54101a8cd9e831c67e806482bc12372b893 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4ff25406 | sha256-f97d8385a0a6719e8c6619e4ee95b4e612c63d5775101371879294b27e77c1b4 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f5f9a75d | sha256-27204d8bf3f1fbbc128a58ae8e7b13f1ac797874e6b1a095b46832d06c090efd |
