# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-91a0633d0820892929ee483cd601c44d030e606ed764348767cd65eaee89c88f`
- fixture hash: `sha256-6d906de02d191088a0de23c25acd9ce0dafee05c1498a2c021d3693ce5ce2c41`
- score hash: `sha256-f33c5af12f476736de835e8d21a4e3cccc818424dc83d1ac045b39e4fd05ed73`
- bundle hash: `sha256-658897c927ace2befa3b9816a5a55902ba917604b4df327ed61b8b38da5c6e35`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-028ea247345f633c6b07542e5aaa8c0bafba6aa7cf71e5143111b89053a70408 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7c11ac6a27bec8a76f40f147951764ff6b17cb3c78f412cf70b3163999e25da8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-787f0c4e08575aa193cc63b008ef3c12c422ea3f6f7af02f2f12674b670ee7cf |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-fc32c241884b3f0e0d7c1f90c8cd31c9477f42afefbc3dcbade8639897cdf421 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5671401a | sha256-2e946720b2643a6dca306a66d5fdd7748e97d5d4fc4b11557d30805d3cb9b8a4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5671401a | sha256-a3e9b3966545b3490a22b9c5336bbe01082c17547d851ab1316ef1362a33765b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-28d85121 | sha256-faeeae31452df550732126dce18266d8d768a1138b8b79522896279533391b9f |
