# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e44a71ffb544349fa10e1154a65bb6e77238a611db5acd86432535b5d68dc4`
- fixture hash: `sha256-3faebbeffb8f05bd64fe046d292ad1b3475373e375c449edb9cff67872d9f497`
- score hash: `sha256-fc7ee798641720621abefa8492a5cebd1c97a084a6c80ff94ff68dfa6fd75587`
- bundle hash: `sha256-12c4be0225863d9e24e532ab34ab24dd0acfe100f7b6f4ec879de732d29a64db`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b418587cdea65dda940f9a601cf2fc169601499e945221393d659c55b40b8049 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-18d462365ee0f8a1811ed2cfa089b5e39b4c07ed31dddca9a17d83aab81b08c8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-35baee881dcd243b140b88486637cdb7eb819c8fe9211511000eebf8314c7081 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-4badd8935a69d98445505c120b133e52da7ef179408feb5231e2198a465e2936 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-96f3e113 | sha256-a9724e6badf9b54247f2fdfed0f74e245a519e473cab4f535fbf021e5e5e7e40 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-96f3e113 | sha256-33d4e766382a6ced500a2a13fc76723ad05503b933cae224fd6ae743986f01d4 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-baf749ea | sha256-79391ee8ad9e1c56f9d35deeb24042dbc47e2bf7e4f9ee18d6c855a170829e4a |
