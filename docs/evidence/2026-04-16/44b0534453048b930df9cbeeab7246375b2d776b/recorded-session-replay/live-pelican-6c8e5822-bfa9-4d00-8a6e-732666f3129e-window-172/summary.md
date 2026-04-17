# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c110d52fa9d814d5415fbbb31a6466d9c241d27a71e256584d4f8da38b7870b3`
- fixture hash: `sha256-7aea2d0a1eb139bffa0a7ec4a62af3e2b3a4882d28c7223058abf7f69edb1954`
- score hash: `sha256-b326c83a4b8ca5f90ddb2589644b454f41a7f761eef55472cfdf089b009bad57`
- bundle hash: `sha256-668fd9fdb30ab0e6d7733be0367adec6aafe29e2bd9f67623a37fcac785a7705`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3c530b06d54c1f577e737bdbbc2ed643a0051ad5929f4a0256034473b3d96cb |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5857037f7876a1d906980e070660a2d30961c1030c9d1bb43c068dbe5e521be2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5ccc0660145d558886d507db122a9bbf3976ab42a4831e875714b3be3aa9a6b2 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6c2d4360d0362556561a37bc58fdbb0f6a6d39a573220980c47ef19de433822e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bae8626a | sha256-f698656c378751fecccc820813b9a7fe727f9377ff3888cd937481b288a9ecec |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bae8626a | sha256-6844d3cbe821103778d22057ab9c36de8748fe14300b54db48149e915e257d5e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-e54001e3 | sha256-2cbe0101dccfe13755802d0dbde1da426fa062159b8d9914fa4a843a2d4baf6c |
