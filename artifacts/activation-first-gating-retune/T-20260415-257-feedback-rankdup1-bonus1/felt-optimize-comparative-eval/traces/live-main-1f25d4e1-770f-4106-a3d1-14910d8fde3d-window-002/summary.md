# Recorded Session Replay Proof Bundle

- trace id: `live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4d327d918fbcc4a369abe7ef4d164f9a5cfa04faf7adb2505c432935e6de9ae6`
- fixture hash: `sha256-6153c43221a8c0bd8b8f42dc9046e70b2d1a03e5bc667d5e9fc62b4aa1f0fcb9`
- score hash: `sha256-07af3e4a2bdb1e5d4dfb48b8ca455209914da7d0e74821af144e5f4f204783a9`
- bundle hash: `sha256-12ec6c3ee90b58da76c2def06565a6ff075e6bf090dfd111cba2e79c35f59a36`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d542a8f0c800204dee4f72d85787e8cb1b923c865594f3befa573eb5cd2d9388 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-6371e55e7bd8309a8f47db14a60ecff55aca6f663510375f807f3702c9efa7ac |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-af1c17676481f23eaf867a487c25f776d702744a06cf45089909c43aea30743d |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-85ccc4c79dbd64f8ab9cb34dd8957e1e90a4f02a8e35765ed62d05f44a7cc485 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-00ed69f1 | sha256-1f6e2d95962511c12105eb19bcb5108790ccb13ba7a1056868837959b0877cc4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-00ed69f1 | sha256-0d1bb334fefe9764a1f28eddf2ed9db4573965b3d9f763c59042aaddfd0f52b9 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-00ed69f1 | sha256-be0dafd87bd9704784fef678c30e1106995f12526eee684ff468db5a11bb371d |
