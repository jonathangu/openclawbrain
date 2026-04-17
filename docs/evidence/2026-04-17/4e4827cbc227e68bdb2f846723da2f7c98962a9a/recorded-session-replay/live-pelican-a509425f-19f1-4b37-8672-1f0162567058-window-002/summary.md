# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-34a774cb3f6c8a06b7737a6a2929058386a540d4a4f6fa06d56dab519cbae33c`
- fixture hash: `sha256-38158baa488957f4efebe2494068936f86320ad50d0d4566b804a6468d20bab5`
- score hash: `sha256-df96b20e38aec40339c7af56ec95337bda19806de9f3004578cb08ac55039cbf`
- bundle hash: `sha256-3026b7ca105cba66eb2af8dfca13f2181e7baf2bafecfeef3db922b14ce31db1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 60 |
| 2 | vector_only | 60 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d2a941461c5687ced5f6be63f00e8602b946e4d86dfa5dfb8e215a577d1b9170 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-052bef99978eb7cef0b8d3fe1b0870ea4e43ea30fc08a7621f50d9c218461aff |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e838bd986d049adcb97ff9e8a1e4a1949c3652fae4b21bd6b3276ea133ebf511 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-ed7e77e94a8b9c180cec8b0d49f33d072a2f0a097112e1799de5ddf70da5dbec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-f93e25cd | sha256-6854a843e7fb66f48d04396b540ad576bbac6c6e7931bf817bb0a48670666318 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f93e25cd | sha256-c4d2128d1008a95b28e6282ec9729bd65f36e5f6f9992bc056fd3446a9591d0f |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-a1edb7e4 | sha256-0b3aa6992f97e00185efcdd474325343b3924ec5200ff7274b8bcb6e5cc85067 |
