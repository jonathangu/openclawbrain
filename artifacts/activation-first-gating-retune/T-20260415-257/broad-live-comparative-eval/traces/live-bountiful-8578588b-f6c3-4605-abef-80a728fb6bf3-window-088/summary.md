# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d408fd5085bee42b21f0981a6a132c6f5610bc4fbc2c34be57ee02be1d61a0ce`
- fixture hash: `sha256-862ded90e7a70c4a33516862a8e1e39d367470070e2af97860bbd4bfdf5f11df`
- score hash: `sha256-129eee97232183aa4da4abd970c7421e8200498b94f16258fe0d2c412881805c`
- bundle hash: `sha256-c8012b171b0bc30a364c470bc768c01983b1ee43337a4250a5789b93633a1c0e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 1 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-608f5e8e26bfe8b9bd3aa5093cf247c400a3d35bb3889f3787f54ae23dbaa484 |
| vector_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-ad38d9769e3659acba7f7f79a2edd8e3e89b3e7de19ea305d13b0c06af0179d3 |
| graph_prior_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-cae76d52c01d68cb55c1fa60e92373f589c8ad306ab249818a30fb08793cb2c8 |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-a17015f12aa137cd12402a5c58c019a7907013b28a4eeb38c0e50c5b5e796b85 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-7edaab49 | sha256-771441073e44c4893e90d7b6c498a16fe866e50984767f324eb6b84f079c8af1 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-7edaab49 | sha256-4fd1cf63641ef39ee4d61f8e7b2e0cd850a2b7ef09d568133085b9c425a66bba |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-7edaab49 | sha256-7d28668b180d9f7a3ee1618c98fb682b466c622740e07f59f6a6e43ec277fbdb |
