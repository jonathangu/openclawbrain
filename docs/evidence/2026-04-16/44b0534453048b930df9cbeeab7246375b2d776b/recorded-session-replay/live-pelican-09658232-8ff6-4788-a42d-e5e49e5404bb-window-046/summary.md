# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3c2d2ba443dbef189a04f697781c21859dae784757f070f6d624a5c22c1fd87e`
- fixture hash: `sha256-d5090234178376a892e6c521a05dfe5104bf688b9e6c7c68cfaf8797d0e0e324`
- score hash: `sha256-18b512dee94125496107875feb4541af8e7283a6614a59d641f29fb39452c9f5`
- bundle hash: `sha256-3574d344db5aaa302ed6d5093bb3377e21db6e339ae90f12c52cf6298549c226`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30cb0b5562af7757589ca1411482395b52af039eced1208652e3e0610a2b0728 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2af7b5ef33723550914cf4c43f518b5d7c77c21112b8c0d541ee02a76310bc7d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-420881b3aad22adf69f7d14087720b1c94b83c0364ce853f299be5cfa856d570 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6f58e486ef50a11b7d2ab595d8b9a5cfb6f7cb2f1da13fef6586d19ad51387bd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-16c1e1db | sha256-d4031f1536eb8edf9a031ed2e70085a350e801882b7ee1d71a7a63c75401695f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-16c1e1db | sha256-eb15b4b721b5b79ac1c1c763f8a380781f3f1d1ef22c8b9c9f208939b1fa8162 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-08b29ea6 | sha256-4ca7952ed70c5bb192f0b5cbd13de0ef31e14bc5edbfe0a68610f18569afce39 |
