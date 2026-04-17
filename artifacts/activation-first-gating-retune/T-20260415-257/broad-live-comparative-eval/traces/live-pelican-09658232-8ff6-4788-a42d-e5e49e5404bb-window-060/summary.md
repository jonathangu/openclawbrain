# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92040fed208ea65585f475c24b64fc03720a2a86a8c84eeb65240f8ebda78b47`
- fixture hash: `sha256-e789097b492b370a3cb207f40a7a3a195c61c7549ef2c7d39a0e569e0dd15633`
- score hash: `sha256-e6042ec516ceff2b121c5ffd3061835ebff4712e477d6a3bb9d2d2e50584fa50`
- bundle hash: `sha256-b7fc7d93d174950aaeebaeb05c8e0fe3bee7523cd580e6998692dc855ac05d7f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-648eaa1db4d7048b0d51fcc33cb635f35068c6efd52393b35ef8355c224bb749 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7bab8dbf0766ff44d65662e7549e46f5f82e7cedd8e05f971994c493d4a7c86a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0d683d63562b8359e0abef3a4886f5e3dc69e1ceca2d3b01bca30cd9aa31a09d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f8c6bb1ba6f7aa0171c83afeb87604f6bd1c7c54ffbb4f14e9b78b610aba3454 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-80b2461f | sha256-aeb3ab7f3002e672005613439ac971ea6082e4c503c9b14e1d41b570ad5bbfc7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-80b2461f | sha256-2f27f30fcde9c31d989d94945e6956bd21509c2bbdf65929e584d66b170b3fca |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9aeded24 | sha256-3b8a173dcd2246d8c3f344d78223f46dbc2cd3e19914b8985ed78642aa601aa2 |
