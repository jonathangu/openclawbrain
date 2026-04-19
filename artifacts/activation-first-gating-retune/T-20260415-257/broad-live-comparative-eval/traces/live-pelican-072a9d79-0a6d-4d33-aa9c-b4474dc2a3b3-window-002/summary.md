# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1df02420998a45ff18b6fa7592e1d6cd553e69e00670f629819f48f156232f3b`
- fixture hash: `sha256-d26e41f8ffbf777f72220318ad80ec7f532c81cc4e8c86beb0f89befd769d272`
- score hash: `sha256-b3a7643f6dfe03cca0f2550594ca3531d20391c4c8d018dce7882ef24477a1fd`
- bundle hash: `sha256-a9dd31523ee330c9ab6c38f539cd8e570688dec1daf9af86ba7b679247d61225`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 1 | 1 |
| graph_prior_only | 1 | 1 | 1 | 1 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-caffae64068969fa7e1d950417642498125ccd7a52b99fe5538a0a0e555ac8a8 |
| vector_only | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 1 | sha256-b5d5cbf3c82eb234b573ea5adc617bd8520dff9f70087ac63cdb3c5825fe876e |
| graph_prior_only | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 1 | sha256-6f9a18ba05356b3379840fd79dc17153f9a30e71aba798f07d84ea4450a9723d |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-5206924812b15e9b7fa4dbb0e711174bbd2c868b16b70036e712a515e75234ea |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | yes | no | pack-0e8d4398 | sha256-25eb848c7611e774addb11275e509c5ff41e12ef2ad3a48ffb93a179d92a584a |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | yes | no | pack-0e8d4398 | sha256-9dfbf56d96a3e8b37608af48d608299dc599f43bcca56688c3ee802258c537cd |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-0e8d4398 | sha256-25eb848c7611e774addb11275e509c5ff41e12ef2ad3a48ffb93a179d92a584a |
