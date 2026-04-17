# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af95153d1f0a3be68251ed9ca1c6eec687f3524276f083ded8a5b5ed5deb8173`
- fixture hash: `sha256-e08d20ecb487c4dc497560c31f0ea6c918c59692d8f02eb17f1047383fc56246`
- score hash: `sha256-74792f1afbddf2e7bbb79cb41579f2a6313c80e7013784299a8ebb606d2db546`
- bundle hash: `sha256-b69e0c2a8fa944cd4dc96f26cdbca26135ae9c19673180fd4662badfbb0c1fa1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af28d7bfd3a7d04b34389f31647ac0f041f3d52e501bb74630c37fbf1936f421 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7cc7a3e004fce6f43c0928f0e1a4bed475c115e164a77292419faf8da7723f5d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-58e00a567f3896c30baf3b67f2783c33dc7d06f3fd4e823d8528a6c479038e74 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-d56d56f5d63baf5c8b39aef112d9ee5229f8038dd7787ed82e9eb341266bd327 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1639e775 | sha256-dce31ba2add48324f9941cdd447143bcfeed59e894ce4c0691ce610926013255 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1639e775 | sha256-87bf53b1a1af1bf32edcfbc88313a57b6715b061788f3423605264f4a8631978 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-05a197f0 | sha256-4cd7fdd4633538ebf2d913c218c2391f62e25dbc752148352d58793f350794d9 |
