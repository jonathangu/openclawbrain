# Recorded Session Replay Proof Bundle

- trace id: `live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e484b4badd2d1a3a3d24ab18ada126ae37897ad6b6cb5ebb205f801adf4b59af`
- fixture hash: `sha256-7081875ca4f0fc3a1b3a1a20287fd5ff9fc1f2b16a465a1e2418cb78ad0e289e`
- score hash: `sha256-7ad0bf2401977a70c3609b73ca62cf68b9913cc60639e471b643b1ed40cd478b`
- bundle hash: `sha256-4516b653731666c85dbc3416996856cd391e300a73ef7f3572dc12a3052af088`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e2ac0d8d192c8a52c6289c0c993dfe551953686d8e0c4d297909e405aea43e25 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e0fdceccf4fb1d73edba11ed690d8f4b0458f0e7213a5af366a164c1732f7a56 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7aa1704d5c1f1fe197eb3b2668b0b365d8bd9dcf1a64d6aaf7b3bcf474a01b7f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f9c91cb1c51e27853c873a05491d97a16b54208c8151f8b72fd01fc014a8194c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5444afd9 | sha256-3fbb6fdf740b62d39f7db3f2e8f422a8037d8831cde386170219c59e3198d5c6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5444afd9 | sha256-3fbb6fdf740b62d39f7db3f2e8f422a8037d8831cde386170219c59e3198d5c6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5043bd6e | sha256-e6adc3d88bf9d8401fe5c27beed6d72f4913c3c42ec4631830c9c908376c4b7f |
