# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b26af2b1e36bf39a5b818412cda88ed6aba582667f9a54ce799e21e291662727`
- fixture hash: `sha256-dc5c60cd5ff0fd0eb8ea43eb629625260e32638ca4678441b2528e3ed52617bf`
- score hash: `sha256-14cda44b8f175d07cb44f4f161d5bf637ddc407d9b8bae055acce5ceccece4e1`
- bundle hash: `sha256-5577204bef58dfbac0b543a08c62dcc5eabcd52b2bad39166ddb81e7142c99d3`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8233e1dc85a16271682dd831a32fd53162f821cae19b4a63ef88dbd637e3c9f6 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e3adcdb40a1dee7a62e22795501ae57bac49d6883b57a7c4c3a0ef7d1319dc02 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-913504bf741c6cee93fe1df9a280fb6354aca0fab29dc5c41e66112f1fe17510 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-e655ae3830aa746b699de285ac7c136bfe80cbe8b877738b943da0459808f819 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e87c6d43 | sha256-9433db60707c16e80490a9f81c9b69155e52741319bc03c6967a4286edb19e20 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e87c6d43 | sha256-6e1b32bf8a42704000f6d6e02380348946154956ed396a780d3f130fdcbd0d03 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e87c6d43 | sha256-9433db60707c16e80490a9f81c9b69155e52741319bc03c6967a4286edb19e20 |
