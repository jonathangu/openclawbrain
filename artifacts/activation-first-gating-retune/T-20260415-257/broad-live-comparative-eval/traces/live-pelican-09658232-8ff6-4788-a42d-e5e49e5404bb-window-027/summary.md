# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c46bcb381e0fc3efac0e09438f41834359285b619fa2a6877dd357e64e821071`
- fixture hash: `sha256-1ef85417b722b0f394a6f903af4947a78bca3d01432416a0cb17a206ec104c37`
- score hash: `sha256-ad4d83d5ea81994191768877b8e34140950dd2cd4187800d7b8bfe2bb48124d4`
- bundle hash: `sha256-4ad65663b533adbe2dbde519029b2ab4f0ecbc067906ed761a70cb06c5d3b0f1`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d30e174da168a75e39ffd3536c03dfe75f2623e328eaa1807c5de3d00819572a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4b67484c09c5ff2a8b4419461e90e02e70c1e0d57b05b45802f4c51543b96875 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8ddfc7c2dd2bcac6839f9d47c9e25533de66f1e1e9de8a4b8b7a21dfb1ae039a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-86470045849a520e7a715e326f0221598d135a11b0f8645b405aa26c1296747a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5495d5a6 | sha256-37f4fdbf84779d0f6e3243c8ca9143f725718da57d72241ab562bacb5fa28523 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5495d5a6 | sha256-0cc183ad404e288261248a38aee9456b1d2b023f7f812007c22fc13272ad905b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-43669a4d | sha256-0d42130d4a08de905943418db03eb012b6d9d73bc700db9eba4b03b2d7547370 |
