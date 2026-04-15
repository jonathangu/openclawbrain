# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a8a2e6a63cc5912fb58030e76267c771c6d07671775935e13384022cf8e7c59`
- fixture hash: `sha256-d3b9199b3d1fba06ec6d727611496f93d92d13e1e28ef25defc3314d0f80c421`
- score hash: `sha256-e6a8ab7a3f8817b91696c5ee9a4e0a0163c03ed88a09802165b6b43463548823`
- bundle hash: `sha256-2e768115c8a0a5437740dcca39b9cc32a056eaa2300b765c6b102cab5cad93a0`

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
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-39cca038bdbd32b11125d0c6fba3b1b3a673e66a982ba05e8a320b541d748401 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-e766b41fa40d75029a66ac6575664828bc784ee9d9ff22fd14a0aa59007f2f2d |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-9443ca412609b0ef0feba88566116f436c7e9b98d0b4849139f91e49889203cd |
| learned_route | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 2 | sha256-1346c37e1f3b70c5699ec33db16ecc4eba21d6539e645b042f927d6c5b3ead7b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-9a959d9e | sha256-fc566a5c9456135ae6fd707ba65337ae8fb144a9a8dd437a626435c21a831564 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-9a959d9e | sha256-5ff28cfe9b3b30df69d4a7479b9fde6e24163fab6c60586d366f83bf432de527 |
| learned_route | turn-1 | 70 | yes | 1/2 | no | no | pack-9a959d9e | sha256-fc566a5c9456135ae6fd707ba65337ae8fb144a9a8dd437a626435c21a831564 |
