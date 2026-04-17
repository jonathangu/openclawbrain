# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92040fed208ea65585f475c24b64fc03720a2a86a8c84eeb65240f8ebda78b47`
- fixture hash: `sha256-e789097b492b370a3cb207f40a7a3a195c61c7549ef2c7d39a0e569e0dd15633`
- score hash: `sha256-22eb418e00a0dcbae8985028c11ee3033674439399e65f36b814fac65092e784`
- bundle hash: `sha256-08fb4280abc93a584e9173fde7922106232d5ae8a2fc94055cf071d0f5c76c13`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e4b8a4f695d7da68f18d00f10d1a98982e18129d8ce8a9f5c933bb7b8c3d881a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-76eb0e85c3694970afac3bae9483c5a887a117d583139f542f6eaf073a19ad29 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-dfa47266de0dbd53f24131fb669f8aac1843fd66be13c4c71428735fc6c6b9a4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2154e3d4 | sha256-7562af7d42ea8ddb7dba7d38dc7750e4e6ad8951c63c91329a46feaf4a445bba |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2154e3d4 | sha256-2c59e2208f47097bc0fd2f6071d930c7013f56425137fbfde1f81669124e5898 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3b908ad9 | sha256-4be7d34d70d6fd3226eb0ddbe0cb1ba36f3aff2d1635a06993680f5e0fe3f846 |
