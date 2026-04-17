# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37ca78c4f79af1f5ccebd457f09d9bd9f0b270ac0d1c7dc3ef10aba20d199a04`
- fixture hash: `sha256-221b36f5e3c3b83dde39237b8133ec3e68acdd74bce0b4e3672a3fac84a8cce9`
- score hash: `sha256-7636a35bc6bb5ccc1c1d78f014ac9d25356b595a670c32080a21116a8a2d9bec`
- bundle hash: `sha256-67e24e65ee79afb5e367bb3805fff901cff513dd23f0c3da5be56d2e8fde19c0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fa6a30b0b756b7163e1ab0f1526218df1fd81b134bd908830d7627bb5155f717 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-404f8bb35f56c13868e8975990efaac41eae31c2f5cd70843f851b42bdde1914 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f762e3755b7cadec7d1cab35b4316dc8e81bbb32d48c3b34ecf6ee9eb0281332 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-9ee4211db0db0426f669b7c6b7843631e67a04cbc928de9167d9199bbc255b96 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4427e1e2 | sha256-9bd62371054b37cdd8c7f8098cdeb061dd740dd2329afc507f9b5f10b41ebb63 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4427e1e2 | sha256-3a0629bbf77ce961259ef197022fed178010c93d652d5641fb04e66a15562f58 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-38647a25 | sha256-820de1c8d8ac61b827ac8fe45bec3b7ad53f9fb807beaf0aa784211050bf1056 |
