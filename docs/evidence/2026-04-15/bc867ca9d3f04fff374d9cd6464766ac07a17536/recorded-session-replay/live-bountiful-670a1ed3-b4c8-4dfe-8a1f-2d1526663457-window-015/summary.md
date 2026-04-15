# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06fea57a3516f2a337e636c80dde5aa0f7b5c4e7b115ef7c15ef4879727a06c9`
- fixture hash: `sha256-0bd1e90ada8a113768901038367ef3359fd513f44e7b3d01e72effd5c2301b57`
- score hash: `sha256-e02869bd0c9fe1bf6e8c91491740877897dc08397645b21912d4b9474a3eb650`
- bundle hash: `sha256-f670513d9cdb763d3ef6ac3eeef343a88733c072fa94338ae2c2cf1b31c66ba4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5c5ee087496e1c83dd50cdb77e530bafbdd0a3348e86d19deb3da1e266821f9a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-855f4d22273dee5841cf5fff4de01e7dfa7546d21eb7d1ee89e19d35f16175da |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c604e4d2a60def588c69061bd1d95c54e71e548f96d0195cfb7b18d915c3e0bb |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3aa0902a704731ea41ab735ea110bc4a08d8ee4900455d08e34f6717e16527a0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2207851b | sha256-3b862fa3c030a1eba5aa29b76c2ae7f375d3c09af2fd333c0a0f32c1351bc393 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2207851b | sha256-41418192d5ce672166450bdf177412a271cf60c5a5f3593de37f677a332ee3e2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-2207851b | sha256-3b862fa3c030a1eba5aa29b76c2ae7f375d3c09af2fd333c0a0f32c1351bc393 |
