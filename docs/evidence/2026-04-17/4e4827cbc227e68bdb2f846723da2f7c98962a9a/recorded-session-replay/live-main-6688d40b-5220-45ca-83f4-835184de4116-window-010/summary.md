# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9af4d3068fe0abcd8b0d002d37c1f3cf1f47d195e7f9302f6c99d8ac1c1ba8d0`
- fixture hash: `sha256-b1358b7f23f888234738e0f7490e996569d9c09e6a59451858fe36290e0374c4`
- score hash: `sha256-334c31d4569ad54ee1924257f4ed21a5c2983155badf08a102ec6669296e5f6b`
- bundle hash: `sha256-0754d246995e11a47ca775ce49d12830d99341d60c2b9f7985faba588904d1e8`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-38fcd5e204f9ce44394f224032307e552dcb6c83cc2ff3a9c8d07b3df48aab19 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-56184110ebd3f872cdc80fb2a84e6adaacd2b29d3aa13a4575ba06dff58f2ae9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e347f5e0fae31091d7620c4378c72a15e5e60734a2780e513d71a7351953e0c0 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-a64164f104bf15ed75766ffa7331ddbe74f2800ab17f0f933f823012191b3b38 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6620381d | sha256-bf12e762b69ec0e74cb121ba4a0db4d7b1a519358cf90586969728176be3590e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6620381d | sha256-6ece5442022fdddcd841044e23d9d21bb78878d72b3ea222360932405664bf10 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-cf96fe26 | sha256-4f4f9660192c7320f5ea84be0f506e57ab5ace94d7468a05cedd1fb587e160cb |
