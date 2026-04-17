# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1e91f891be11ad983e343a9bbb8eb7e094a3203fdeb0cba32d80844dcceadc5b`
- fixture hash: `sha256-c962d7bf59f91132e81f529b35b43a46128d3cc144f19a803783e383eb2588e0`
- score hash: `sha256-f84050d0626b2507fd2fa24b00d222c72930796755accbc9ad8ace5ad0b8ad05`
- bundle hash: `sha256-c07f11e0c16488950580f14272303860abe17e171264bb531e1edaca5c7d6eec`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5abc89ba1c4aafac24d8b492241ea58c50f7925494e6166e3016c9a753e61584 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b174b7ef60371b59d96f9f22624eac47ebeeb082f2209e30022db9fd33b542e1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-04c4bc299095a5bece0398e4d90d612e98405e38330caca119d403d86dc5da30 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-144c62012cd042dd517ff7b1751e07d92f3700d987f6f61d289d9aa76dc17bdf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5bf06cca | sha256-c2e72ead50fd4a0d60262cc1fb2f370f4be1661fe998bddbb3b018d647cf3263 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5bf06cca | sha256-6959edd3e588461a328eabfd6adeb23601c50cad9ce8294b9f477edb5c1d294b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-67689c5f | sha256-8b8c2d267c6205c5dd2f861a7bf8ca94bab33e2ab65ca8b0cf845c721d2fc4bf |
