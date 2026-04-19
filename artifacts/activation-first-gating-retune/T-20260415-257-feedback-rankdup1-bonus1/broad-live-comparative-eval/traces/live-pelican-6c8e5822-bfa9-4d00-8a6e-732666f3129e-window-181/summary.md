# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50cbfb4de0d67c0910ccd1f15acc9132454b767d6a9ef6092fa51c701d086751`
- fixture hash: `sha256-ed982aae33c06dfcffb629c09975a63d396b69570ab9ad349366a4a66aa757f2`
- score hash: `sha256-ed62e59a77fcacca4a91eb3dea79413f0b4b92e5a8007c18a1c21f64e9dbaf0a`
- bundle hash: `sha256-eca6f1159c324288791b2041ebf6d9bbddbf16864467288fbc1d6c3ab317e581`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 1 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9ce8acab11edc00f581b930ddd46ccaeed311548b8f75f0398d0e21fa5078567 |
| vector_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-9c54152372d148cad410f784091270c2fba738ef88f99250f48c97b8f3d8eb3f |
| graph_prior_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-43c4a5c3979471fa5a1ce7ad2ba7be2140432cad62d488e29dff105d0d6dfb7a |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-bcfe168203737af00d6551254f5d1df1296c18f9b75be01079e8c146d4f86a0a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-e2d7a8b5 | sha256-a654aa07801f3cbe163b66690008f2b4877c8d114d3a2f4b9d35d0e1f366d3a1 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-e2d7a8b5 | sha256-6e2fbe8bd6798240a7bfb8659d6395b82d883a3565406a1d4d856ca99513a2da |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-e2d7a8b5 | sha256-a654aa07801f3cbe163b66690008f2b4877c8d114d3a2f4b9d35d0e1f366d3a1 |
