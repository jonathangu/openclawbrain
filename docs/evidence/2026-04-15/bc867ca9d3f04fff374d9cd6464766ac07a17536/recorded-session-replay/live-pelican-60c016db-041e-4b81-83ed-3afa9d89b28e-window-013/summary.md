# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e2a057c58ceb7779d689dcd4238dfbc3207e352fc341de03ac7a06d504301da`
- fixture hash: `sha256-737f6561e785d3bc05d3981f983d5cf16785ca63d2f46199fbc1baaeee1f2b69`
- score hash: `sha256-d944ec4c01ddd7839554cd2aade69a3672e9dd187609c0276362a3e84fa2003f`
- bundle hash: `sha256-eb0e8e5e9c7a4dd254ca09fb6c77b158d0426bb9a602a2eb6e5b46ca9a9067e4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1e7a79c157dc055e3ad83a213c22e42badb5ac82b3ed30aa50ada887959b805f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1a8ce99aa618872749a88d4086fd25e9b4f8159451cf691bc86e38ffaf74a3a8 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-128c8650b8f2dd9c29eba1cbc9926c576b44cc3d5f936bf049198fa318b1e2ff |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b62b6e3d5ef93aa93ebc945bbd19a1e6fd106fe064f76cf3f9c6d94aa0e15ea4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8fe3db06 | sha256-dcfb6d0c76599221c6f1e3cdc9c9071748b53d90fe2049c2d313a7ec5a9c573f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8fe3db06 | sha256-d7b1eb1b2884c7a313af2503988299626b280d928d646167f89b5fb59773f48d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8fe3db06 | sha256-dcfb6d0c76599221c6f1e3cdc9c9071748b53d90fe2049c2d313a7ec5a9c573f |
