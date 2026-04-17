# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dd29f792fbfbc606fc0ae81485babcea7a498bf8f85b66de2333918434925117`
- fixture hash: `sha256-a84a33537b8d24e458443c5c6b1cbd9d02b490a8b56c8f49f8509184e51ddc87`
- score hash: `sha256-2ad3d4ae17602ecd63ad00390c2bce7ca5fd3f1c7ec6ec4ed495bf7f6b19a631`
- bundle hash: `sha256-da4a1b6daff4dad1845444aca84e46cc2989fb0f44487bf8f4ea9a7cf1598fad`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d6181c2140140f5786a710721c2d0cc92976577da480a328a542e8b790bc4990 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-66bfd3bc170ff5d65b1638a47215a40fc9dbb63255d65d5a6301804525408bbe |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81a78c718f5d0d2aaf4d356f154521733c49f12f6cb897d1e2b0e2f0a29c9009 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e2777ae23077cd588414c439a52200211b35dc1b2378bf405dc4e136b5564fec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-927fe6a7 | sha256-7349c60c4a4d2dc1e2597c89914de717a56cf62e47ca39d6a004cde85d5e2089 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-927fe6a7 | sha256-6abe45bdf298ed2fe549d8df9511dabc2211285cdaa2ab6e5d53db7bb863f135 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8f7f3dd4 | sha256-d3e7c041a119760d8add323234f2350f48896e4939d375adf354207e7bb3d9c2 |
