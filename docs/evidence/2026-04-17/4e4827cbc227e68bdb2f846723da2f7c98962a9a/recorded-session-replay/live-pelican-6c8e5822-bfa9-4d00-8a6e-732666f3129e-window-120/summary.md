# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67782e30fe5f9982125f26c2ecd77317f6b86c34b8443a476ff968e4172fc9ad`
- fixture hash: `sha256-3275c723fd5e55770c99a0a3826bd67e0749405b630c9523de493fe0719c674f`
- score hash: `sha256-6ca7a7608bf08ad854859c9aa4165a27bc3078378e8e69ba2a9a3387c49dada5`
- bundle hash: `sha256-174de60f78458b381ed1d636c2394df863d44dd536fbeb654cacc9b6665c9105`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6faa741f27c297696cddf75c51e07e62f9d376795b5d33f012fd6c625e199a2d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fbfb39202deca01e0b9c2749ed6ac5b7f9ac50ec8a6cc4a23dc12c6f191ae64c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-db391e7b7c81f18eff2193452b67f3955223f549a9daf50b0d83e47140d1084a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a173424082c6aaa9ac354abc89f2661fc52fdda533cce031c1da29a29f6185c0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bde5d0ea | sha256-a27bb388d328fede7e0e3f54a40da24bd4059e76295adb21289030ccc87b4d4b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bde5d0ea | sha256-179d7cc9f04e0fd2a9ccc074ae7f43e9acf32cf9b8c2758aa475484fb26f8327 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a3151bd1 | sha256-4c537a3456b2e42ce34815f37c4c805b32239626c327513174e14abb25cb7cb2 |
