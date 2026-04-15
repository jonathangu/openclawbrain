# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d5c4c3a45a443f8773663b6b718260034f858fe43001100c3e44deaa92dae64`
- fixture hash: `sha256-8ee17d6b70fb97105471476aa616629c3b433fcacd6e10fa09857f62252427e6`
- score hash: `sha256-4897b0115d1a6eaa793d1d95a67c8aa4749bf6f7d08ccc95439dafffa2b1b15d`
- bundle hash: `sha256-91fe5c810efe8b15049a0f93f0b1b6b6c6a1d08bb99b7204a562aea50a28c0b7`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-09e55df402125c6b04d503b2df670ff995850f1c31d072adf7d8fb44788c9b43 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5ceb700cf24fdda9b544f03f4f41ded8b4edba533c19541ca20462666db1500d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ffc3dd8a4dce98999ae018e843375f430284d3b9e0f7aadd11ad04cee8c68079 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-f3b1d80cdf6302b9880b0de0833f7d9831f7db3fd4c68825bb58df008ab63b6d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4f0c7ab9 | sha256-d168bae5b2d06eb58733270d61bdfbcbb542e1754e6aff487170ba7d08757812 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4f0c7ab9 | sha256-a128b3441e7e0e273723e804d33dd185d66076a4dd33404a7895a14adbf55716 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-4f0c7ab9 | sha256-d168bae5b2d06eb58733270d61bdfbcbb542e1754e6aff487170ba7d08757812 |
