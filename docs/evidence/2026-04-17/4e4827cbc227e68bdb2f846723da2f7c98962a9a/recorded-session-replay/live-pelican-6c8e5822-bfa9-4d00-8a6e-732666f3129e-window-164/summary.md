# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f337daaeed6bc47fc68765c0195f530bb4ce38ec076e00ac4c73412b426d85da`
- fixture hash: `sha256-5e5cab708ce5b294bac69d34a6279b47e648ad8d40ed85f35998caca6e589c7b`
- score hash: `sha256-e6e48d12c33279a134cada11c85a612d66ef748ea458dc9643f8225e7ae07641`
- bundle hash: `sha256-6107f95aa9bd51a419d1e656b86e6467cbbad60e78eea7743b4476c8c834e92d`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-100c819b269094add6922ee0aca0d157fd41366c476c3703f8d276f1431d3315 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0435fa9c29ffe1d20b47b77392a09258b07862032bb0a64efc44c24b0b984626 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9d4ace4aea4d3d91901623ed901cfe9cb2eb3122cedb210e6ced412dd02e5825 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-96e56e57f60947310bf9f7b0c2fcf0084c668c939bbc9ba617a6b9b5b680476f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d54bc2a3 | sha256-a4eecc16d9c35207455f07d2e3591c7ff9f7fcb7a9e1f5aa781e09cb7c834375 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d54bc2a3 | sha256-0530a32486c9882c6fe3337cf2c0606ed1d3a934a72e11d676b0610d46c0df06 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-5ff4e3b2 | sha256-f54db751806f232aea7f59531e0548726dbb6cc59777209394ae628e1f5f1b0a |
