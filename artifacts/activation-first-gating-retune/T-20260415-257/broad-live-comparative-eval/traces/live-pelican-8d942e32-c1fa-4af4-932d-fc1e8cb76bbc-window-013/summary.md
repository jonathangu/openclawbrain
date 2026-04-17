# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e8abe8bd791e7d6cf823eab880acb642edafbee61d1547309c32e0509f5a12fd`
- fixture hash: `sha256-55ffe1baff231052090ba7af248a8c8c581b0ed9688d4757d7043a08a2fcb4de`
- score hash: `sha256-aa4ce0f182ec30565e6424da391f0a3d6a950adf0122fa4b21d6de806335053d`
- bundle hash: `sha256-d849260ac1c578922a8c52e10ccc3224dc766d1a57d2122abe0676d075a80ba6`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1da03994a3e5454931ba1a5c62fc1691a06d32d29326ec5baedfa4f4b490d130 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a32f4e77b7c148eafe61b28d397b10684dce9f2931d3319195ee0225c89ebb42 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ea3af1f3873138065e0ac4ca2aba0ebc14a97a55c25aed470080894531f61274 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-26b951c2ece1e674b346a6d165226197c8275f26bfff8cfceea1046e6086985e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b607d2dd | sha256-46c7e8e743c54826ca709f6dd33ad64c735b3d956cac190a955998976edb505f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b607d2dd | sha256-a7f86191af425b3aea285916cd524a2f3a76fa3950053a5defa772d39184d275 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-c8efbb54 | sha256-8314a107b054b69b4b9abb08e0fd3379eaf143a2722af4b5a17e91c8aed1e46b |
