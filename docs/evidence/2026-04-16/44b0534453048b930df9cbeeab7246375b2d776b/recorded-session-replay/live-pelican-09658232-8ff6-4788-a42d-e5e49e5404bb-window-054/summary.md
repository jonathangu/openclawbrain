# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f8bd6e98ba409d4b92ff33d315c90158dc9f7928f49ee95918b29862594fc07`
- fixture hash: `sha256-f2d0f492e33718dcda5e95309dd8b8ae83d2a012ce623b86565c773255e59638`
- score hash: `sha256-6550f7be5ef1c47993ba1dfc8c713f266d161241ebac42e7d0f4fd9c550fee45`
- bundle hash: `sha256-a935fa574a0dfdbe4318240a9965af3defba877a7244bbb4282b2adbc12e4c86`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3a49035e9fd3e0717342039595aabed753c46d3f982a6fbdc847832f0114d10f |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f7d7414d19a0643ecce3cc590502579b424ca8c2f2346dc4da6c751d529eab69 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-95b89117fcc76a1abaf938bd382de0b35a6bc2c9506078ff8b7537754c58b874 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f9333133a1540641005a80a7a7278a9c72a397472014ae7b0621ff937497497c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8bd1794c | sha256-35c91a1df4e6506bbbc5fe8c7365f65cf72d142969970b29330ba5e731e0f989 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8bd1794c | sha256-f785a68b639ea241846306e3ac1422f5decf8f48697e5a61aa825934d8b0893b |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-372cf4dd | sha256-22c28e95ee757a4388582b071f8a9cc5c7f1f23bfbbe188416f8af3e2296c2d8 |
