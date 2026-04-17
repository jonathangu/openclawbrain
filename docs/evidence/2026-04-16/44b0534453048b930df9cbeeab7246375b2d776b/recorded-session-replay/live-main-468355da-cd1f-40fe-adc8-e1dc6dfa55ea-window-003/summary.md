# Recorded Session Replay Proof Bundle

- trace id: `live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e484b4badd2d1a3a3d24ab18ada126ae37897ad6b6cb5ebb205f801adf4b59af`
- fixture hash: `sha256-7081875ca4f0fc3a1b3a1a20287fd5ff9fc1f2b16a465a1e2418cb78ad0e289e`
- score hash: `sha256-8c9d7cbbd57b81ce72bba1b4b2bc44919cd08ced3c73d1d48911878b8f093524`
- bundle hash: `sha256-8ed7b6ea4965f762ae04968dcf62e09aa0892a97616f7d86e6be236d4387018e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e2ac0d8d192c8a52c6289c0c993dfe551953686d8e0c4d297909e405aea43e25 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-40e0a1d1c8a98d5fccd3567146c1e96dd3878c52d9d745c056a50315a9845c83 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7095b5b30327fb7a20a68eadbaa3c1fcf88e9895f90b041c08365b83b3758f7e |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-9c6ffdc7d423c3426b37b4d3807457c90594a5a8ad0cc79a1e1e2177a6bfce55 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-aefb49b9 | sha256-b477a5152f48ff6dcc09ebe0ce1f87c99f607ba998b526b8b1865e7e47069d1f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-aefb49b9 | sha256-b477a5152f48ff6dcc09ebe0ce1f87c99f607ba998b526b8b1865e7e47069d1f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-aafa574e | sha256-1003c86a812b0f5930cf42840e9eacf0185f2c896acc0d367155247060c1b15f |
