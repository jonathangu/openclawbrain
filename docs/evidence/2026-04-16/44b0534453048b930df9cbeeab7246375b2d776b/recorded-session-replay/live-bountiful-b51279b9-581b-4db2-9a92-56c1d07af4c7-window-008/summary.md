# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e05f1753c6926f7421800b8c31b26225feb7d252c56da262406bfbe6a5f19442`
- fixture hash: `sha256-dfb3a653195516c3dcfd429e73c49db1db57d8e9ca226f19c9bbf361b6ec9f1e`
- score hash: `sha256-9f862e6b15320c3a12949469b7787cbf949cc310bc8bb7847bf16588ca00a345`
- bundle hash: `sha256-de4009ee0635ad40b0f557f0f4b12804b95552f5a9245c62fc7be45fa5e4bc4d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-560b7797ded1325b1e9e670019b80d08d942ad37f4650ec817919f6cee20dcc0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2757a9e92496869bf35e9030b37735a4e5765961240681e18ab6d12741157338 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-55543dbcc3cf66ea07b52bf69e9fd33279c3cb36695906a6e5a00c4fc2d64aa8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ce050647fcdc249821f037738332bb97baad1bda93657921a14df93104da42a8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5f433f91 | sha256-b8887529aa7a7da3133d61f97850d1dfbafa0e6c1077afc1f850150ed5787efb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5f433f91 | sha256-88d1a4d7ea5995f5d8e6ab6949d3e2df3301bd90eef6ea3337564dc8f07245c1 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e4188736 | sha256-2e6100da56880973d7ad1e7db94d06bdcbd07841a07a95084262a2f8cda22e16 |
