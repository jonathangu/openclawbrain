# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033`
- winner mode: `graph_prior_only`
- trace hash: `sha256-694cf444538867e625d49591796eda7824a3f9914c6d50782ffa8d2751091f0e`
- fixture hash: `sha256-cdb2b18e3a901c8928c86a3e5d6789c9de0d594dce56653b0cb654624b8e744f`
- score hash: `sha256-af3f7df12761d6fc8f45a6ba08daf58f746db012682094a9f3bb22c954dcc276`
- bundle hash: `sha256-3d5741bc7f15dfaf9790b99a80d77071b439884bd81af2535d9ecbcc94fae065`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5406b4db2619fd299a4dff36fb17ece03d149828bde2ae07870bc2e0cc31ba06 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b937c0075f16951f8e5d01dfea008ab646e15f12e395dab889491c808956c72a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8d2af24ab5e20391a4bf0a0aafab327897287603439cb6fb6c41e5302d2756f3 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-50cc8fe28907b8134e999fbfba8138c63532d777d54a29ae9ca9666e147f913d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-226358b4 | sha256-205f4468bc09c3b6dc7fac33ca1c722f39cfd95a29894b42bb601025153e2215 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-226358b4 | sha256-f11933d97753b49e06c3a91a8a955d92280f5d9c39ae853a4ea489e4b3121805 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-226358b4 | sha256-205f4468bc09c3b6dc7fac33ca1c722f39cfd95a29894b42bb601025153e2215 |
