# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d5c4c3a45a443f8773663b6b718260034f858fe43001100c3e44deaa92dae64`
- fixture hash: `sha256-8ee17d6b70fb97105471476aa616629c3b433fcacd6e10fa09857f62252427e6`
- score hash: `sha256-9407ae1d6428256e4f602a719ba0a0b00dcf5797d5ca7292e0623022a46ff0a6`
- bundle hash: `sha256-666417039078368e6cde4841213c1c594a2c1977c549d34a257137f65337c76e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-09e55df402125c6b04d503b2df670ff995850f1c31d072adf7d8fb44788c9b43 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5279a091ae1cad4c53636c2e6811f92fcc2d8f87da2969ff17fd277ea2fb64db |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b17a690524d5976f8849a668801eaf23d0c0cc2a5f78336d4e97289ce972c940 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f087fef20ef7f8f632205cb188315bb1be61bff3f802616825fc126fb6ec9362 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0a2a49bb | sha256-35fbdb52807f5ee69a3874262f218d602f6c632b3b0a73308af7ae6bb5439494 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0a2a49bb | sha256-8e01b7e974c4c9fe2b1f301e2077fe2360590e9e8c5d0dd8dbee6dd886554ec4 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-85903010 | sha256-f74e85704f2deb7381a121d185761c3af5b024446cf0b30fa706b0903784bf17 |
