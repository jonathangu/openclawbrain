# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-59b36a090be7bc29212f1b29aa7bc29b12f23e5a450aaf05d1c3eab4e44abc8c`
- fixture hash: `sha256-712180e16240a5850bc7f0f166cbbaa035f07312fa10c3e98606123034cbbf4c`
- score hash: `sha256-b9513052675ebb592be91bd27d630f17ddd00a4817f172f3ade08009e8fc30c2`
- bundle hash: `sha256-4e2ddcc1ab630f3cc5304190682c835b5f4710681968b3a6eb7a7b2b8dba1e84`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b108d912f44bc4b15526ab7db40ee75e868952f9ef4952b3a83ae96ae65d4c1 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4194f3bdfa8350fcb22c5c11ce0b9ee7b7d17f605d201d57634f1e89f2ee3d9f |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-0d0eeb18d4e86d047e6539aa10d3c1cdaf391ca61f20d17105445d9963060d0d |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-53368b0ecbb49416ae0b9a5a6fb058640401df4d6d8d8a848b4b251d657711d8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-a1d53341 | sha256-1efbb008b7fa8f3b6e7b48cd2faeb6d93a1bd75df56e99671354ffba675d4b3e |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-a1d53341 | sha256-7742756d9f8d1881be59968a71038378fdf8ab2d38fd3bf7c9a9bf027f0e26fa |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-a1d53341 | sha256-1efbb008b7fa8f3b6e7b48cd2faeb6d93a1bd75df56e99671354ffba675d4b3e |
