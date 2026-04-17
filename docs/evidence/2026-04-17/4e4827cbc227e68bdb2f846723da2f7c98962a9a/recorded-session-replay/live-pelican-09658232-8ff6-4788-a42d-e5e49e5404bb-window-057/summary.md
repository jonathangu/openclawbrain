# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2f2e67ba6e9f3ee34d9a729b960d4347b90c5776b36c8bb01215597777ac63b8`
- fixture hash: `sha256-31116913aa40fd67b6f1a05c1b62a0f72f8a386379a84cc5c256525c2b570370`
- score hash: `sha256-e71c31fc581b21145fa1d9f50edc90f75bc580a7663053cf63a4df9d5198f29d`
- bundle hash: `sha256-86674ad80075fc8772f3b8ad22cbd8e2d530a1da834ce5aebb8c4252b62cdf18`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13aa42d069a6fbba4caba9f912ef9cadf19ea12093ab266f931b4282b9e22bf |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-92eccdd882435fbd981c6e6a0466f3a49c58fa4b482261714eacf31c963e2a71 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-61a894919786648f138fffa2e326214c0de5820abdb2c996db126d950ea8de56 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-17b8f1aec2cf6abf22558630b6729f955fe72190a2930a9c20bc324c31078c70 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-afedd231 | sha256-289f169abbd56af90eab62a2933035909e8dca6303e9041536daed2ec55a5ef3 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-afedd231 | sha256-519c02b0c331743b1ae614e123fdc167294f52ac662add22f9555fd3829001ab |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-72ddcb14 | sha256-bed65e3ed194215a1fba9ab9ee6d6f3cb042f00430cc1d3bc9d5f11d1319abc1 |
