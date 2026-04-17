# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bc492042fe348d82c21faa673d938d0577346ee06614b38f34d614d883fe125`
- fixture hash: `sha256-166379a9c9e98e60de3e148d45fed20846d7dac8b779bfce9e0299ba405d4f98`
- score hash: `sha256-a94ba9a987ad3c569210094b2e074b94b9ddb8ee83c09dff47be2ac0a96b3f14`
- bundle hash: `sha256-0f87ce14dd8d9fec12e6b0685499faaeccf022e951a5479812037da76fa68b9b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df7544610c7c12f9cdd0d8aad84f983991755b60031939cec1112c0295581782 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-809f1c134efe5ebb0d094bcdbf419b93c7623e9d0cbd3989060fc3c8aa424793 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe8ca44fe7b4f4a6c98ee8b5479e89148df28dfba9fef8df41dd0c6e4f7ac21b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-71bbadf71df51ada9c2ab3415f1b31f1e25fd893303def8be99f5be7ff137805 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6fdaf937 | sha256-84355914a9b2b50f2c353a67c590c6445ba2b60c0905933d30c838198e33cdce |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6fdaf937 | sha256-14a3b379315dd9eafbac698e101c3ea4205fbacbab3fd59f3fb7c07da72857a5 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-3330ab94 | sha256-f182ff0f5080099330dca13fd9ec938a3be9b238bdd8d755058bb691839f974c |
