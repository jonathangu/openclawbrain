# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-91a0633d0820892929ee483cd601c44d030e606ed764348767cd65eaee89c88f`
- fixture hash: `sha256-6d906de02d191088a0de23c25acd9ce0dafee05c1498a2c021d3693ce5ce2c41`
- score hash: `sha256-9e8f7f237f967fd91d05101ae3d169d4120f8e55c8c67fa3baed8ef0046c4bd1`
- bundle hash: `sha256-6f90b113e2484982edf61c6ee0a5e171077a7a46ddadc24668fed34b6639ef3f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-028ea247345f633c6b07542e5aaa8c0bafba6aa7cf71e5143111b89053a70408 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-310206025bb62feafecfeb48859d20e98d7f0d03e79f43a3121c31e4e9a39322 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ac2da97316ddbd5e11663f1dd3b259d9b4c4e5a4d715048d777389d3f401a011 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-124ed1061ed12985439dd2187433d8e24292e9b98bad558483d878775e1e3d9f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-961ff7ba | sha256-7847a37bcc9b57fc4953dab5d694a345f955e9546354faba54ee77e89ef0b60c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-961ff7ba | sha256-c24dd42bd7a2c5cca555a53323cb1e12b1feced8edcea01a8ef355d2e53cd2d1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-688708c1 | sha256-cbd48c10c4b5c8ca200bae16a7cf4d9f272c221523126087ba6fe7c264a4e938 |
