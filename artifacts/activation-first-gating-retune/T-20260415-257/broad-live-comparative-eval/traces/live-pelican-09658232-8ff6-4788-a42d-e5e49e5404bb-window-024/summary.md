# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68581f69a97780aac278954522193e99993d4befdc39acceb8ff881974cc0178`
- fixture hash: `sha256-d2931cc864933b7e6af27eb1382872e22dbe9358020b6cefacd8fc78d2489792`
- score hash: `sha256-c5c05808a2f5a3d0668666804b3ddb0ae914a792c40afeab0f4f3966f7437071`
- bundle hash: `sha256-fc8fff6c27c39d30e605f5a344c83b3ebb4a253b24f107008b83e89ed2ea7ba1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-181208f7b843fa2c39286593bf1b96c7f44d97e1cb317cd9b55efb3be3bcccb4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2f91242d87abdd03201841f93b38f27e1781ff5e793e868ae1e81f1b25cdcddf |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-304c0b01340a81b464aab95f4e528dc1a10db59be174d6da1049bbb478113471 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-fb86fda75905008edf5e9f37d5a1c99b93c12d233cd2136462aaf665abba94ff |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-47531fee | sha256-7a114b1c183cc8605e836992d5cbb357220d547275c9b50a5e30806bb51aa9f0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-47531fee | sha256-a9d73a592aeecd217dc66c4bdd9274d0ce6a4e49e931ddef9173ce22d767dce1 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8be9b94f | sha256-f8b6f2dc1594342f6fdc555157ed679967d5109035635f8b519698af25f8e9c7 |
