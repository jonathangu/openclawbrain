# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a7deacd5889e0ad224da79a8b0dc2045294d6fcfe7af40c942dc92f39f797429`
- fixture hash: `sha256-c533c8f4dffd9730ef38a51c383960cee8574f83f672136e59013a1d92400c07`
- score hash: `sha256-6e6a9bfb6cdc6e08232cbd87d73432c2cd7429b9e7d3397019e102dea8e2d144`
- bundle hash: `sha256-860021b2cebd314938320bdcef44ef40ea025165e274b94c93d43807ba69331f`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9bb519c15cd685af618227f7aaa909dbc79bba57d56f1160ff54bd9389dfd5a8 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a7328bb049fa55ee9b27f7a623abd30ec68ce9457c0d3243f611e3c798701dda |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-c6601f5ddb110094ea9469d5024218bd3b90de75efc17f98750ee8a7b0e22ca4 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2c1e6bed38a261dc493f89a7edaf433a70a00951c05abf45eb3c2166c32cc4d9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-16a7c022 | sha256-0a9e9e38064ac0ff2b90a0db9af89617aa66371ebf1999e784dab8c148821c4e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-16a7c022 | sha256-94de3f967fa433d64fa7a1f61c46129b12c3a9e947587998e134c017af92dc92 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-16a7c022 | sha256-0a9e9e38064ac0ff2b90a0db9af89617aa66371ebf1999e784dab8c148821c4e |
