# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2684bc9ce52da3283e7e269a65aeadfa9bb4bda12e0a5937bb82b4e7e3f59ace`
- fixture hash: `sha256-4296b198ad4b2382e867baff61985bd607aae4ddc54e4c60ef5ccb597fc35e68`
- score hash: `sha256-462a481fa6e1497b5041d9ac075247a5ba49725d30d2e45f289d6ff81f1ff960`
- bundle hash: `sha256-983326ba98c9bd70ae5eda45e84915802218cd19349f0557c4f45df5acf8fd7e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d2479b1210d374fe06946cec83ff362b307da973ce6e0c46c380449deb18879 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fec714c96b60395d08337e82d5f2e8a30311f75c1693509e1f6b1882f539747 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bfe27141affe6ba638f96396494adb8cea5eccdb9acdf4ca28fcc84b57939b2f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1f21ece059e9298e07e5c060080e284d739e6648ce58350025ea53fd7feff972 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-452a2773 | sha256-dbc1419efa2221a208955081c60432b681b7fd51d14de40318b0d343501c74a3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-452a2773 | sha256-497a393c8836b458d398682752b2474156a95240ee179d926514c2db7f9a92b3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d6fdee38 | sha256-ae2d6a0a2b302c45c31195061f5dbe6502715776d2f4f57d0ab2ac0230188bd1 |
