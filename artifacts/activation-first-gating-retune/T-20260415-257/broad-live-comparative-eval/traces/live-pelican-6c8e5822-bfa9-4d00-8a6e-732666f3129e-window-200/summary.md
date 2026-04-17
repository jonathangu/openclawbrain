# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200`
- winner mode: `graph_prior_only`
- trace hash: `sha256-438b689c90e1516f117c130a44f955ebe5121f19131ef3c8af4f3b72e782a392`
- fixture hash: `sha256-fef64d4e61173927de1b8c7e42759f7ee5918ab3e67738573626a046f39d5b5e`
- score hash: `sha256-52d2387bf1c3523d53dc75c49fe6f413ebc3d546a15a4b18ec94c35968a853ea`
- bundle hash: `sha256-60fcf2d3577d1e4b7aad8a2c55cbf7511060001252e2bd71cf2f983a753a1b74`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f3e6c6ba4308832d244620436e1eb71e4969051bd02e8a257e4c9a12dea8653e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-92c107a93e595f51c64ac9b22fd6f422bc04f4d52307a624a9b618a3acc75534 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3c0032102d11ef50c8dcc056004f688a3f142e0a4eb2017b35cd12e92f75919 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1b2125a6f8994f36847691b95edb01f97ae62be929f170f52167e5c47e2de6fa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-32485515 | sha256-5367c2fcb14321aad9116cb49776cde76e742b3e5ba743178f00c91620b88af2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-32485515 | sha256-a7b26703ccbfd67d0f8e51228e1ddfa51567d315b3f9e17335b2d094ff7c38b2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9c65db5c | sha256-22ac9524133a6099209695cd23b01cee2d4b8cfd160c4dcecf9d1f6893888d67 |
