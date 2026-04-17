# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2f2e67ba6e9f3ee34d9a729b960d4347b90c5776b36c8bb01215597777ac63b8`
- fixture hash: `sha256-31116913aa40fd67b6f1a05c1b62a0f72f8a386379a84cc5c256525c2b570370`
- score hash: `sha256-8ca93240f7b68bc72724369fc91c98b1eb25da1d87677e9af56178906ea8da3b`
- bundle hash: `sha256-4f7e643d5ec184b2933db47f4d786ac8e45f9af5ab71beb082058e390b018753`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13aa42d069a6fbba4caba9f912ef9cadf19ea12093ab266f931b4282b9e22bf |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-53bbdd6393a02d843dae2af1becbecf00f3a3928365a62e1155e87b39b58c73c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f98446c820d041f9a430b54dd8ab1dc1f770c96148aa6da9f22090bb8d52fa1a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-e1fc8f8032db8a4954d3dbba5bbb2405161d24239ded687c488006aa5ac9367e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-86f8d45a | sha256-fe8815d4b4debc6f23dc701cb2add8afd09f65a2a52f7b3cbebd08b9c441504e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-86f8d45a | sha256-20727cca8841294dca1d759f8a2b206ee0765cd40e091c0e05b23fcc6f08dd30 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-49e8cd3d | sha256-30e55a701bdd123fefa0283636e17fa65e4f14c2d3179c63e695dce5c51d938a |
