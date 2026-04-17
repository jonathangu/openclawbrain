# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d5c4c3a45a443f8773663b6b718260034f858fe43001100c3e44deaa92dae64`
- fixture hash: `sha256-8ee17d6b70fb97105471476aa616629c3b433fcacd6e10fa09857f62252427e6`
- score hash: `sha256-e3a457959d95bbf63d1ca571e72b3e2be538c63cfc1b7b78041f007d6f19a506`
- bundle hash: `sha256-0a9991647a8f63f5bcfe70dc172b592ba1693b1157a522caead3ca0649b19e48`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e73d72910eec5afa7895ce560843f7e44eadc5eaf580315a6b771d87a31eba7d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-273348a4043eff3381fe91128573f68924cb50b1a4c6f0b1d0bebcb052bcde59 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-a65a0e12af6791852a66e1b241e7cc4ce9020e125a33334071cbe4ef641e41ba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f529b044 | sha256-4d180a5f364f41e76b98ec41ba59be91be51e3ed436b5403428145ebd3ab7318 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f529b044 | sha256-85ecd37a24d7aaeab1acff4200bff2d16536db4bc8c0b1362dd00b86f1455ba1 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-708f9699 | sha256-3c6ca816039ea9eca9e46a7bd076e8886eb4f018f5a2c6f404d8c2a25840e749 |
