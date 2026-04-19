# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-91a0633d0820892929ee483cd601c44d030e606ed764348767cd65eaee89c88f`
- fixture hash: `sha256-6d906de02d191088a0de23c25acd9ce0dafee05c1498a2c021d3693ce5ce2c41`
- score hash: `sha256-c5580121775b93d60da2d58e32aca502bd1363160780095cf9156a38e04b231a`
- bundle hash: `sha256-05cbfc3b0649ea0d2fbeeef5dc20970f02c2dcad45a9990bf8fcd0a2f095659f`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-a28b44c62195c17fe4617ba89f72abec83995ddc9f21da5cacf85cbf7efdc02b |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-ad56f4164e920bb1d98ceea53649dda37a1a2b705a8312f6f7c981b72815f5d9 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-d17d533d34bed76827219d8226ab8b671bc28d71c7f862c6ab8a4d0cc74f3930 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-5abc7b30 | sha256-78466b48e9fffc07f30bd8d462ec19959d835ecf98ead1c523a7e1f1b14def64 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-5abc7b30 | sha256-f45018f90e49c4c67344bcc472d69eab8bb69b1c851f5e79c68a791157006391 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-5abc7b30 | sha256-78466b48e9fffc07f30bd8d462ec19959d835ecf98ead1c523a7e1f1b14def64 |
