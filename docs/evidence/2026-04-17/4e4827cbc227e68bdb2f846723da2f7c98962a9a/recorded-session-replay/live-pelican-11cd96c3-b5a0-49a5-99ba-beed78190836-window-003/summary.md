# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ec605c7758b471d35c95979aeb2cdfe7a4674e948b05ffbdd6046eabf723431`
- fixture hash: `sha256-076f85a33a3de7d14b01739ce6654a252ec79b49aa247d0b8cb77da6c5a8a9ec`
- score hash: `sha256-3a63bb78d918152728fe37dc821d6fa517188c0892b26f0e2d20a3ffb79d0012`
- bundle hash: `sha256-081e9663ee7d45dd0657100db461ee6359fa231682ee69616852a3ec07896a99`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b4e2e11e992d3b83a5df7a249ce0dd37bdac79f45db7926d41f83ea82d964f78 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0a0dedb5a67b02ba0e3f5104cc476c6fd4152adde53f33f4180dee397ecb882b |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-94d195d2aa98380a7ffdb1fba2a01409b22d48c881425d22a327091825401cfe |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-454c851e8996f449c5fc1d549851be4d5229fb4833fe4bf1bc36c1eff3502194 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2125afd7 | sha256-0566d84efa49b063bf24040d388d0012e6bf14fc95a68362f17aba7ed573445d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2125afd7 | sha256-6d55c7aad113489b770be0387ab1a08bd128170d3f2f05526d61a226e1cfc018 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-71fae986 | sha256-d6bb51f5b2711ec215bd65c7218bcc7ba62a627d9be0b1dd179f6f53e3aa6c97 |
