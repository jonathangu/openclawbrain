# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8836c19286d7dfc7e25365f74f5e0786007d5f48b08d8fcfba5fe79b0f03c2c`
- fixture hash: `sha256-0ffaff36365448396a5594a68d8364ec6eacdae9fdbcb2693a4ddbea65547f4c`
- score hash: `sha256-73c1ca2747deab446d42398e02441c38270035180af99a8fcf7b830790a74300`
- bundle hash: `sha256-0a2b23c3d4a90519cc5b8d7eb75aa7b1243d9ec42ad2da6992453bf9e80628db`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6c4e67449219060e0eaa53a64e9ca0f2f7168ec707e126564ccb072cf633b7d0 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-7cee34514a15b0e059c7c35caa328e765f73ef29ec18c9861cddeecc4f8efc78 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a3a7da41125193eb048bfe3be00866b0c97c2c429625a2c39ab7268b0a2dda65 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c982ef40045d5def35843bbad40bf50ff3f5ff050e3637e3b84d847d3dfeed05 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-bd4f3460 | sha256-f30985e403add40d02c43cbba8aaa4a46273ab4054462be604ae721afb9341fb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-bd4f3460 | sha256-d936c26dfe306e05d4b3481d099eb33d9e4d8d44efd2f4d67f377afa4f5ae0a7 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-bd4f3460 | sha256-f30985e403add40d02c43cbba8aaa4a46273ab4054462be604ae721afb9341fb |
