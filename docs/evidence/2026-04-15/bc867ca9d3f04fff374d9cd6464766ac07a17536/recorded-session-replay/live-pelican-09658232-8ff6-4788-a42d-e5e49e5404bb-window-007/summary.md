# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-945c2e59668de577944ec7fc5dd5f9442c630538679d6020f6fdac64e2a21a17`
- fixture hash: `sha256-19b533ee2cadb7bef94e2f868a3d98284f247e98f26920f7fea15136681e3d11`
- score hash: `sha256-0ab4f8ea9e27a0019e704a601017ef2c2dea890ae6393b0f37e29a592d8e7288`
- bundle hash: `sha256-1d13271642cdb09c62f136ba144bc29affec3f2b4d0b2507876d30cdd2df03b8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d37994d9aa92d2e1c7fd5cea54b3093f268f662f580c5608c088fa86597acbc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-247008aeee5a0c3c81a6a463e449f540c09814122dbd11dc12e256fb5203a2d1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-72c99bb9462cd46584ed395703b12fcf36ab846d506a24d3d04da483454aede9 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b0018ae6137cb840b099fea01bc979ba4842a2dea1dd7177316ad0f726a840cc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-01595453 | sha256-9ea69e3f3d0ba86b5832162486acb7561b0cf24255828e0508adf24fa59aa566 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-01595453 | sha256-e934a6024ea81d96f84bd3f14c3baa514033bd084115d4f3fd937f78923abfc1 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-01595453 | sha256-9ea69e3f3d0ba86b5832162486acb7561b0cf24255828e0508adf24fa59aa566 |
