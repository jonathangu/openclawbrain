# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-14ade459ea986baa6e4e71bbbde0e89dc1fae7980400ac765d36815dff4c4f35`
- fixture hash: `sha256-9c30c978d165bf9a25e14aa9b77d9a12a45f7a9014b4a8204bd05ec1ae139d4a`
- score hash: `sha256-b5df5da0b4b8c4759b8511b8a298b19b7f1454fd80c4ce44942e688909412c24`
- bundle hash: `sha256-4ad05f97bb2c5f3f7d290c722e5638e08086db2a11acaa02abc1675fedf37e55`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-334c135bfa30ec156738872f694abf9297995f829f0e8e1c5041f315be0a98b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-51df8d248745cd5511f96458aa66b6781db9a33795bd7bfef0df6247c3d95775 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8a2790839cca5f08448cbef7558e08ef06cef630f993cd394e7e3a090986eeab |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ba95eb5aa4eb71659966134c712a72d049b5846fe8981592e7fd825115161dd8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5bee40e4 | sha256-6b7fcba686357db02e21be042fccd135a4a4a3df4788d8ec47585533eb9e0541 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5bee40e4 | sha256-2bdf3ff13576c29f242c4ce8edf6783e413b700a2d3a5b863b2ad32898a9a89d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9bc42d67 | sha256-478950b2d272a71774c95c8f9322fa9ff0b2a989ba65d419c790a7a0446de628 |
