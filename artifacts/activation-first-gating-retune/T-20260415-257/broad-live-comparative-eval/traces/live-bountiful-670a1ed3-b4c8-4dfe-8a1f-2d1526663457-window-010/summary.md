# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c80a3decbe06cbf3c4af187d8a5af847ce341540f23d409b6e7d63d31df4bcc4`
- fixture hash: `sha256-741cbfbe2c3d2f3a4ab8e97bf7b8405a7d1cec581f3191dded735c7802b1e00f`
- score hash: `sha256-e60a3353ee5d06eaafd62cb64959ef97f2fd3599d39bbed8d706964fecdd0580`
- bundle hash: `sha256-401b92d30fc497ed490af149e6de9939d1a92fdd7dcdf90be6ec618b596ff624`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16c0b7f4b283cf0cadf9518aed3354f26372dc3c9867fbbccefe14e243137800 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cf81bd3ea8c2720cd04102d01c0581312510cfee94a3b43e98176a04e196a425 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe67cab90643f31238834309a6532a759669c42fc8ef2a45811ee6b4a07e1986 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-011b9140b0014aefdad9fc7329747c4eb8651bc1c0c304fa71ec57929cf303c6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f32866ac | sha256-ea82226ec0098c35965319982ee72fe018841f3d38e5125be78f2d2065003985 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f32866ac | sha256-aa5501e7b8c63aa686509d4a7dc2eb925db7eddb9d0f55f4d69e3a8fedeffe31 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-07fcc5fd | sha256-c4b5dd9ed999c6b85dc80ecc1fcf965728fcf90b90e1bf68d77634d53e8de25f |
