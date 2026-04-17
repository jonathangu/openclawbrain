# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f07ed34ccc6a5c54819d12a1e93195c70560e32cc80e0d0e09592b4765b8105`
- fixture hash: `sha256-00d9f388b90351cc79a6666fb1faf09e6f2109bf7c85e8cdc18048263ccb39a6`
- score hash: `sha256-e5fe5b389ec2b1352c4d56677cd084fe211eafcdc19b0c8b4b47d9339c474546`
- bundle hash: `sha256-f53be3aea7418ce9ceb289d32101ab8f8d5c22b5fc25d457031e32413ecffedf`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f9e400fcbe43d9ab55b6048a20689714c3c7aae22f85e1babf49f3474335a32 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cda61152947fe7dadc06e381127179e2c5e550b2a711c76875b89cf294c3dd9c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-898267ecc8619e31a2660d5f6abf26bcc0c43c362afa5801093284ea7612bce0 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-e567dbff244ee9259c3d62b8785f1303693ced7e8b238c294fb247f7b99e4306 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a6599c16 | sha256-875db18f5813334637b2edfa9a6b62138c1195f3291530dd823e1c2154a37f39 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a6599c16 | sha256-93ac557ec1d6f68696b124c147b95eb936bcbb832bf661ce901f8e7b276bdab2 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-1238afb9 | sha256-f081bf97d7f7e41075feb1cad9902f6ff71b24640fc2c832af2d5da8a92286e1 |
