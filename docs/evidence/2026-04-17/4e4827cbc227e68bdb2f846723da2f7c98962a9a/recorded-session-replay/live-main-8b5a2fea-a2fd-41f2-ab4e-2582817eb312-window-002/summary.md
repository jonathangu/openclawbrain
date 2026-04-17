# Recorded Session Replay Proof Bundle

- trace id: `live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-e0e56ffd1c26d20085e7a9eb3248f58dfab8c43d92d6bc35e804da203ef4f7d9`
- fixture hash: `sha256-e4b8d39277cb985d3e9ee559f9e373775182720bfc10b6d9350141f9c5016460`
- score hash: `sha256-41c83cc1b104d934f4ff88ef87b057e858c0e086acb4407330852279de0f4e63`
- bundle hash: `sha256-69e2f0daac6a810b41f3b6dd75554d6d067014bd551f85c14da42d517a8773dc`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 80 |
| 2 | vector_only | 80 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 4/12
- phrase hit rate: 0.333333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0bdf6c0bfdc77dfb35df2ddd80b080b8e6bbd2f8f1020fedbea4770e769e1c72 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-484de9884695b5a724f485c48172770f0a1bc279349afc3b1e56630a4bc70cad |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-67d913d31bcbbade690bb660ffa2ac45d059f408aa3eb8337cc1b5c0b4d1fccd |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-d2388431dfccc67835e8423de6f1d69ea491e478f6c4bb10108e92c398dff3f0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-96d29652 | sha256-5530b8fed3880e5e2a1ad35a0046254924653f145d11cb98a61a22068272f0d5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-96d29652 | sha256-688f874cc57c2c92c6d491c40954b0fa8037eccc1d469d63110766ea3a965e6c |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-fcfd5d0b | sha256-3f740459a945c14d4e602054078a3d2e2e75113cd70953a715b9ab195b763ba1 |
