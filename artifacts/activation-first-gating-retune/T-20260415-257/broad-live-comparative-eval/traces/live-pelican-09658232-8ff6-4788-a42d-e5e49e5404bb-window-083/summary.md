# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083`
- winner mode: `graph_prior_only`
- trace hash: `sha256-84b9a4843680de911479c2420a8592984c3d84b3d54d06debdc96d5c918ea030`
- fixture hash: `sha256-be373dad3e692162d5000f12580f9371232c68a9b0f09d3136130b3fe2a640e9`
- score hash: `sha256-8cb6aff5f0c8a29c58e8885752edc6508f4f9e52bd6b83d3a778ca475728a350`
- bundle hash: `sha256-8f80ad22a60557040c3641195b42b65c3903060a878e1f7e8642f451cc214301`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-34864147a65f338d5fe87baff27e70ea8462feed84ac2fbd4644ab5e3e006364 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2d3b4cc8456ab17f27a69d10906d6a98c414140d3389735f64cb5d610a5c1335 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2b756bbab93b78f110a2004fc84ee9b1b3202f858d5b116603bbbc578425ea2c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-769bfb26b295518effca838da6b2d2e07b2c6b46e90700611e95bef3503d5a92 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bea8aa96 | sha256-04f945e9a65d9e924ec143544fb097cec939023e4a9026e0861d06f086d12f90 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bea8aa96 | sha256-970c78d92e85b9938ebcd3a993d8cd651fb7d4de80d7df0fc60501897b51bd8b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-71fed8d9 | sha256-d226fadf55846d1df015e3558ff3dd175a30b9264028564e2600a1083fa538f9 |
