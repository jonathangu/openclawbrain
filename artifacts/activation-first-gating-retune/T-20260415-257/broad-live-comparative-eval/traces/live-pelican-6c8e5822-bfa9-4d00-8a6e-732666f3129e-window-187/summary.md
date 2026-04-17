# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eae72c8906ce053ade6bf66b6f03ddc87f48a19f8e1b50fd6f47ba9774ecb440`
- fixture hash: `sha256-80de414b90b70f70f1d2f2daf70e3430dc27d1af7b593fd0e1e1dfcb61676ead`
- score hash: `sha256-55c2f65a733604c703876adf60f4bfee1d9c4a6880f21146ed1e54ea9de1f6fa`
- bundle hash: `sha256-51099d3abc158a1b8482227e45b3277ecbdce5ce8a87988bd4e79d8236656666`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0e2afd5c5e27e893dea21e62c6e8b163bef7241aac8748bba68a4d993b31b8a4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-119e805e806857f0feef8035bd8c02def7e432a8027d3a4056844b9b90dc9af5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d0fd6362ecb6fcb99d01569b9ee0e112dc1a3da1af2d2fc073d0c3f2881258a5 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-1c318e34442c3ff6ee758ac2af890b7f6b6f8d6ddcbef774f489105ba7684ef7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d67c911a | sha256-283c52d5229c6f7ee90da4b1d3b74fbf14c4dfff68fb5491e9dc541ba097e43b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d67c911a | sha256-ffedab6410bc3ec8c9312ce1dbefc9248d2c9d5116fef6b1020a2db1fa56cf48 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7fbad5bb | sha256-67458b9d60ac3be43d30aeaad9c8ee57d39accbaa5a1c606cd386ec53c16683f |
