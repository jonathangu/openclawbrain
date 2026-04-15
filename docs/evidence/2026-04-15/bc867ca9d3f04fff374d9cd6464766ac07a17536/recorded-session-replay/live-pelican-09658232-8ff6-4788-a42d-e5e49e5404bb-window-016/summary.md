# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0e0ac143317067e59f64740cdb9f819c48d2981153767f573c0e73b22b2b7c81`
- fixture hash: `sha256-dbbac8f5cf8c52842e2689d4f90634fa33bc0bae1bc0d3bfd9ad2ad85d720253`
- score hash: `sha256-62a5cbb17dc7e864c8afb05e4b56deb070b62bd80f767a581142302ddbaa82b1`
- bundle hash: `sha256-706e36d3fbd8b40ce27078ed1cf507cafbc62bd4ba32bd2d4c6cfaeb7ec5c465`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-745541acfd3bce8c03c831feeecff054c455963b939319f1092513f43c7bfc25 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-e4fc427421fab6b009856dfbcad67a89f8b9d0d71485404507c29dcc35300054 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-d5618a8281028002e4edf1d826e06a866e0cb64c9862811621b94052dd7b161b |
| learned_route | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 2 | sha256-78a4cec81e7734a58007eb12e040030dbd0a05a7d2d9b9d0238488a1ccefa1cb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-85d742ec | sha256-7098979ee86b81627306ca3bf14b48cc3fc0a8e5060d86ac504ba6aa572cc4f7 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-85d742ec | sha256-49a9c6b0af0c697d00f2bdf2669532f74b4a897d83ebf6eac10e314e9246dcc5 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | no | pack-85d742ec | sha256-7098979ee86b81627306ca3bf14b48cc3fc0a8e5060d86ac504ba6aa572cc4f7 |
