# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-91a0633d0820892929ee483cd601c44d030e606ed764348767cd65eaee89c88f`
- fixture hash: `sha256-6d906de02d191088a0de23c25acd9ce0dafee05c1498a2c021d3693ce5ce2c41`
- score hash: `sha256-f66bcfdb676ff8a252e736f6f114b4559b769f81b25d6677d4fcb2347875ef1a`
- bundle hash: `sha256-0ca34101f67b1a2f525e0c50f129f6bb14be65a711f0e5473184aad38998b328`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a12c588e0489d875f17afc174236c1d6c3904b317e5a079c50850ee7fa28f233 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-150e899ed05b92861f1910871b838ec58d56f0046666a65041be7afe0de6842c |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ea79099aed653ddf12916178007050ad85f47bf99fb1f895d4192c61e7e183b3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e0ee539a | sha256-dfcd88e450f39866b8ad725b46d847c17ac4eecf37f1562192c5b63d91307a08 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e0ee539a | sha256-28aef4226bdae79b63a768d6b36a0b57e9e058b118a9ee36e9fec90e66c9f9ea |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b35564a1 | sha256-394fa5d2f60d16d8ed67182f065ee870495bfab056a3bb910878a4e78ee580d7 |
