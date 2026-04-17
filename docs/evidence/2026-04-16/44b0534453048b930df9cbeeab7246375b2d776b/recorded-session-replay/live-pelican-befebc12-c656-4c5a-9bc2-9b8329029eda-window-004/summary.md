# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-974c0caac77d24b74750a03083f2fe960327dbab94f044b1f352645b0c8977ef`
- fixture hash: `sha256-5332e2f75d9b84ce32dc4225385441cf4e1fdff1345733b1234e8eeb65449d9c`
- score hash: `sha256-63f1f1001be313949efea8b2f2de0f63446a0fd49c28f726345282fe605500ad`
- bundle hash: `sha256-fb8e6fa91c1fcd0001aedee2a2791c3fb9231a885988336c75b96fe64edbcedc`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-dbc3f90d1fd6323456b57dfe2268d1b7eda59d985039bca3aa485da45edbfc64 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-81b8b1401ef5bf548939490debde7c14b88cc03d36ee7be33a1b00054f1d39b1 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-256ea214270d9d13b82270cb69f256188c3d9e8d054ddfcbec550fb2ab8f0423 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-cd47b7136e6f9fa2dc32ba7e2f54a0f7317db7737427b58f64d9db0c80614628 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e877fc51 | sha256-c699bb27a3de724a704041b3c45b2e1753d7b9b5ce1bb10cd035d404e7685b83 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e877fc51 | sha256-f01c68750be671f2d0809f6dfe0dcf218776567e4ede0ddb649ee5ee2f920e1b |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-18506dce | sha256-c2f7f2f410eab492d6b37c9a16f69c61aecb7a04fff90d723bae4ed6a922b888 |
