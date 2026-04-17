# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1c2fee7fd4eb0c2720a3ba15050df8108cb036feca9a01fcd35c4b07aae7a9f5`
- fixture hash: `sha256-61f4419f55eaa7d0c0ca68a6f768711b70a4823f4e0fe058cff8927193ee8afc`
- score hash: `sha256-98bd5ce4eb1b6b7600355e0bd3ee9bb51d2e1ae6dba053725b5bf1dc417160d9`
- bundle hash: `sha256-2dbb9f622d013b830eff2d32628040cb5398f0873f947db34faa92142f0b4666`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3dcc36ea1001cff13b10454b28af88c47e797eba5193d74b4990d61c1caa8eeb |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-6b205091854279133216ca6402bd755379f7a9bf9483fcc246fd92943dd0354d |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-582f6e8ce49b88c86fc7aa7a17e6c3eaee6621c46f59bc10390bc6052075bd1e |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-09829b8448ffa1c69b7e1e0b76e8f35c8918452b32ae19a12f3cfa123ab8e18d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-171a2e55 | sha256-00c9477d9694f1a87963d45add7845ba5181a9919af630c8088eafe353f1235e |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-171a2e55 | sha256-41a375030d4d5407f47e25cf1c1d62cb00b3a92c2a7c20088f1e378492b2d926 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-0e134352 | sha256-ac2f310c96f6fe992d2e816cbcdc0b7575f975ca51ee8cf145d64b6da7938db2 |
