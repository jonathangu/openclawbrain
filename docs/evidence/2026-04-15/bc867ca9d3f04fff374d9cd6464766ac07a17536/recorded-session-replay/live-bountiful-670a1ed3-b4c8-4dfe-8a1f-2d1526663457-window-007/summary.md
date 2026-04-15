# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-829093e68c8b369222680a9ce88928380b9f7729d760e59ece8cf8d1e776b82a`
- fixture hash: `sha256-31ba926396eebecc30aa75781e7d614cd75f3d45744f5fc68d2426d0829db138`
- score hash: `sha256-5b09a90a9e1bccdf24d3f8de2354fbc6afafcb6533b4399efd6b8aa08057547a`
- bundle hash: `sha256-cf3fc4c4aff680a41ff254f61cc5f8270a9db346706d4bdf84f9c1ecfda4bdea`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8d38776b0b580dc292e4970ed98776136ff0d2acc01ecbb7a8d527a0c51a84c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f6fcfee542d814c509fc6f652858a09e4b4904e37aa2678a689d879aed6c210b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff3fafe3a655c1f3138165e415d9dd61841df88fbfc8bbe5a93e338265039c6e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-95681024e7a22b355c2d535a1dc7fee43154b3537e90f03511f004c9c28dbc6e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-68dc88f9 | sha256-e6e1f91b8fa98834e609a552cdedd81e67e575116ebd550b927a68650bee956e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-68dc88f9 | sha256-bda238dc36249f9f0a7f3e75128a6decca572d47c83a786b3781e5c12cd9db33 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-68dc88f9 | sha256-e6e1f91b8fa98834e609a552cdedd81e67e575116ebd550b927a68650bee956e |
