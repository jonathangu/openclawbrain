# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af0623cd896f3d36aa832764b91c449eb65a56e502af4829ad2995082aa19cee`
- fixture hash: `sha256-729b2a143706d45b443dec7a409dfdba222ee805edd97aecb9fe78e30ae910a9`
- score hash: `sha256-426f8d2644611f1e69f33ab66b48e99c06cfbde1875600a0ea8b444cbaf9d143`
- bundle hash: `sha256-4a3b2b1c30c63237bc61ba184c2cb61b549d3bd5441b34cd33da483d7a2969fd`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-72dc7dab3dc434226257b098b5889b33f6d9a175c84b5a7ecf9e06dde7b7bf77 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a255cdf6818b88a0f438d65ed9356afb944b247da224600026ebbc9c07829e3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e5f186d78712bb2ddd1a69aa41952134e682848a0fd4638f3da996f552727a69 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6b0ef22860decd59c456707cf432658a4ca4d22fd58e965b1763d0b6b8abceba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8d9caf20 | sha256-a6f4f7f075ab0008b60f29284441cbbe3dd98667c27402448381afecf5c22571 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8d9caf20 | sha256-7042c3efffc840f594e6cabeabfb3cafd645e1739a26a9c5674e9f7332403e7d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b806712d | sha256-f885e9bd2570df147b7e77928b7eb2d537fc9aaf5b9b9c3c637e9de1ad5a0983 |
