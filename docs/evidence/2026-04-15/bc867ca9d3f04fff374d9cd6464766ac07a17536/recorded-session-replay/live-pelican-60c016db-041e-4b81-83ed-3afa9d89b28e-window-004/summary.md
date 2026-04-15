# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98f498b917834ee9c0a78d5b62a338d5c94ab2df87cb501ae8615cf42d07619a`
- fixture hash: `sha256-54eb8df766feda2c6211a7171b884e66a0008ed710f7d28bcb6341bc861e92a9`
- score hash: `sha256-b4ed03b62bf014818f386770397d235fcbcc1ee1f9c1c4b7a7f8d864116cdcf8`
- bundle hash: `sha256-d49ad6af95bd7d842d3bc9fdab2b2648e0e844dadab6cf5337ab20595650f5c4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6670af2aca1f1e71cbf3c0f145ce7f96dddb89bb0330719aa7609642a8108f9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fddf3227cf1ec430c6a6248828fa07f07d164920f58c3df05ac6d071f430a1ea |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-67d64a74bc0e4e4062f185273bc23b7d9c271acaec380abc6bc5146daac56d6a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7087a4ee77ac3b1be6a59b75c1a1629f73724bee6539fe243e3452d06e9ba5f5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-559907a2 | sha256-528fe46dfbb2761434f926c825763717ef846395ba5c9e36c208ab900bb5f379 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-559907a2 | sha256-3f2f1a5cc851f7c7148ed61595472900d05ba647814d1591e8d2738accb1a741 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-559907a2 | sha256-528fe46dfbb2761434f926c825763717ef846395ba5c9e36c208ab900bb5f379 |
