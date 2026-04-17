# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a8a2e6a63cc5912fb58030e76267c771c6d07671775935e13384022cf8e7c59`
- fixture hash: `sha256-d3b9199b3d1fba06ec6d727611496f93d92d13e1e28ef25defc3314d0f80c421`
- score hash: `sha256-bad4df646120bffc4dc9359d8825ab5a8466c7fc825de8ea9d6e9e6b6864e787`
- bundle hash: `sha256-6e2a71952b8e6fa22c620bdf812631acfddcb23cc1272cbac1fcb672cf8a6dc7`

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
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-4eb7f04deaa71197dd33fa0e460d7201e5003fca96f64b7624dbc7b137dec7ab |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-13a38588d8f1ac4341ef2106f9d94237863ba4cb6e4f412ffbf36768cd2c8e92 |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-4ead810c4bd5f44d97e0f0f9c895d8ac0e7eccff07b3dbcea2fe994531edcf9f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-20637ac3 | sha256-76b11ae8eabad5a024f23dc3fdabeb86b72e62e5eec47dcc08d048eb8a35d703 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-20637ac3 | sha256-f99b1f3c91570b28944a19214c853eaf0f57413169a49ab45d08ea1d6cd1cba4 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-1cce5d56 | sha256-b7cfa370b291f212d2e887c7e266fb4d3c085583b845b1ef81d68bc05fbda4ed |
