# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67782e30fe5f9982125f26c2ecd77317f6b86c34b8443a476ff968e4172fc9ad`
- fixture hash: `sha256-3275c723fd5e55770c99a0a3826bd67e0749405b630c9523de493fe0719c674f`
- score hash: `sha256-3c8600eef198fa6798c07c662b6aa9ec4efeb0f104e741eb5eaf5f872c7eb56e`
- bundle hash: `sha256-4dbd90bcb7444f18dda24fceeb5332d3ccdaf23e9b1eb59ba653272dde14a7ba`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6faa741f27c297696cddf75c51e07e62f9d376795b5d33f012fd6c625e199a2d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a5f126dfcedecf4add1df3c80328c73d6dffad028bd1778f597015a9605c5950 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9ab9ca7bf43355123330d9bcbbf3d450ec5b2d860bfb8003782c24e25f17319 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3c3979d48e782411aa45fc8d19d24f20a134081a1045c6f710cd80b0f3de0e49 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d110fc59 | sha256-b6551ff8069b343fa450393209c667280029fd226ed5837ad63c1b30886da94a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d110fc59 | sha256-5ab8254e8c87861c7d29a69b895a5ffade700df97d05eb3cc753e53a97021c88 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b6404740 | sha256-a8dc741b5cff2b7191cd213bbc4311a5d2d4d04abed8adff27c5637a5fa2f429 |
