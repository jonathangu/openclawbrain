# Recorded Session Replay Proof Bundle

- trace id: `live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e484b4badd2d1a3a3d24ab18ada126ae37897ad6b6cb5ebb205f801adf4b59af`
- fixture hash: `sha256-7081875ca4f0fc3a1b3a1a20287fd5ff9fc1f2b16a465a1e2418cb78ad0e289e`
- score hash: `sha256-5baa670987a96038a27df4abdf0b076b60718110ab9a97b924d8e7af9aaeb562`
- bundle hash: `sha256-e35505e7a71831a0aa5523a44dbdff041e6d9b4531da30845a3fe3a22232dfbe`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e2ac0d8d192c8a52c6289c0c993dfe551953686d8e0c4d297909e405aea43e25 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-22b1a7555b80f1f79a2d645e3fa32646b3ff0dc8c77318a26c59153bd03a59c4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a99fc27e9c0cdb2954d514da47a28f296c6e1354178ee39fe5240cdf6ebfd0d3 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1a5ea720ac5af8961fcd843a0898f43dfb44e2f12574baa2f30e8e2c9e88c9ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-83c8ab1f | sha256-1eca2d6e4eab30a0d638970628fa4706ebda14dd74d8e4793c7ef79b7030b751 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-83c8ab1f | sha256-1eca2d6e4eab30a0d638970628fa4706ebda14dd74d8e4793c7ef79b7030b751 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7fc7b8b4 | sha256-9748c2e6b56afe429782be2bd8a9567a3ffdb559bed88981705e99a60ea2c3f3 |
