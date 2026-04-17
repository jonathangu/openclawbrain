# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-59b36a090be7bc29212f1b29aa7bc29b12f23e5a450aaf05d1c3eab4e44abc8c`
- fixture hash: `sha256-712180e16240a5850bc7f0f166cbbaa035f07312fa10c3e98606123034cbbf4c`
- score hash: `sha256-21d801e7df1540b1463953922ae6d4e8a91883c3965d1ef62985f10d4dbe0026`
- bundle hash: `sha256-e64f44c7ba93eb7ae57d5a577feb1e7f565657b1bec73fc21da310290a9a9f67`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b108d912f44bc4b15526ab7db40ee75e868952f9ef4952b3a83ae96ae65d4c1 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-f2f1a2500b80c256438ac2113ee36a37d30b256bea344cbc5fa3072e05828248 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-d0b7c2f6d5487ec4d3f43c9b07500bb1caa3270a9e6efde5c3230e82b89dd5d1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4b9c3812c62f678ddf1224658986c1fa4d79638d1f6ac6f2ec9690afa0ebe05a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-c7115982 | sha256-4cdb6781cc6c2e36aa9f75155e71c4d7ce53bda1b9763a17e5a401089c3d3438 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-c7115982 | sha256-9e85d81c89fe256eae57519add6ee2098410644060ef16de7205429afa40c334 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-66c22a89 | sha256-e769ee9457977392ff9105d9857fc959fc483d9897faa8887bfa0a1a04a5ca2c |
