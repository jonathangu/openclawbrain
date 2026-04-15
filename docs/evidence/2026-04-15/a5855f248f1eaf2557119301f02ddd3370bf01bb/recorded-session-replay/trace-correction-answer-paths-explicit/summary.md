# Recorded Session Replay Proof Bundle

- trace id: `trace-correction-answer-paths-explicit`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7675563cf859465dbd888bb06d4b86fedbd83e541945091d1e8df5ee12e84c1d`
- fixture hash: `sha256-57a7a6e1a6991f696a856f4fea90684928d5f3ee0f026ea6b5951d4fc10cf426`
- score hash: `sha256-76e39b8d10d22257e3caadc56360dc3e33587ac6fc3ce7a05beb5cceaaef6114`
- bundle hash: `sha256-d41ffad1ab6aa7928acd4c969a780bf1cb05eb985a2ec0bfdcd287a3dbde1aa2`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/3 | 0 | 0 | 2 | 1 | 0 | sha256-2042863833d6ce7ab296d4ac789fc38e8c68b0974c203443b4c7c040ba9c0cb6 |
| vector_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-c590c4f6545e10eb0ec8b57cf3eabff5e52acb70273b9a4a4e1265b248797241 |
| graph_prior_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-db539afcffd95471af2a48a1d4f32771d94de9a6c9d3384586ab32707ded2659 |
| learned_route | 2 | 2 | 3/3 | 1 | 1 | 2 | 1 | 0 | sha256-d796ba2143c571b847a8848313ea35b796b9bf3f2032bb8f17159cdc38c47aa8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | explicit-paths-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | explicit-paths-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | explicit-paths-turn-1 | 100 | yes | 1/1 | no | no | pack-bd4f283a | sha256-4184826960527d05b0c6f9f650bc4d13baef725d2521ca4ba4b98c41192a8c58 |
| vector_only | explicit-paths-turn-2 | 100 | yes | 2/2 | no | no | pack-bd4f283a | sha256-4184826960527d05b0c6f9f650bc4d13baef725d2521ca4ba4b98c41192a8c58 |
| graph_prior_only | explicit-paths-turn-1 | 100 | yes | 1/1 | no | no | pack-bd4f283a | sha256-4184826960527d05b0c6f9f650bc4d13baef725d2521ca4ba4b98c41192a8c58 |
| graph_prior_only | explicit-paths-turn-2 | 100 | yes | 2/2 | no | no | pack-bd4f283a | sha256-4184826960527d05b0c6f9f650bc4d13baef725d2521ca4ba4b98c41192a8c58 |
| learned_route | explicit-paths-turn-1 | 100 | yes | 1/1 | no | yes | pack-bd4f283a | sha256-4184826960527d05b0c6f9f650bc4d13baef725d2521ca4ba4b98c41192a8c58 |
| learned_route | explicit-paths-turn-2 | 100 | yes | 2/2 | yes | no | pack-551495d5 | sha256-ce88b26fc00844cd2384b43d7c743b8df2ba3b0494a782baa8562e6ccefe351a |
