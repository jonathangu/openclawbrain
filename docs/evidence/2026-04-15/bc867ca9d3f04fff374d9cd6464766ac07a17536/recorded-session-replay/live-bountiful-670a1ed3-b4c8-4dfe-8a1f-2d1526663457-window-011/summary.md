# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2d785c91c6c2597c88bfdefe91898000c30733ecf3cca8e1fa5fd2d6621049e4`
- fixture hash: `sha256-63b7942b83cea800c5fc9cb957ce0307322538d9d8e1a745ea7ab80b74e65911`
- score hash: `sha256-0af5cf6bc5bf2787a711c25676d57347a6f783c72dc8af05619bc08b3723ad14`
- bundle hash: `sha256-70a1164dfb979ef593d8f5014cba167dfd6b3b815564307835d45c6356761a5c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4648c104e20ab98d8928f41590949536cf65a6240f7fac95811ce6126bd169f5 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-cd48b239e869c6b7c1de3abe27d319aa25e9c039a0d2673c0f64e5c1a9e6e9f9 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-94d7ff16846c75c84e649d7a287f58e00405da06b2212b2d6473aa680ac726e8 |
| learned_route | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 2 | sha256-1e93b72e958580cdf5b55ec0f0fadce7c8306fb3d6566cf7325fdf4f9f536ac6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-78a95ed2 | sha256-80a6344b07e29f89eeaa79ca1a2f73972186af9a4ec70351f092bef6b3359f22 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-78a95ed2 | sha256-569a972b1dfb9f530f26de54a9c43be0248bd5f6e7e322f8d833466373574c54 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | no | pack-78a95ed2 | sha256-80a6344b07e29f89eeaa79ca1a2f73972186af9a4ec70351f092bef6b3359f22 |
