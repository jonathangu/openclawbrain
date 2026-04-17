# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-34a774cb3f6c8a06b7737a6a2929058386a540d4a4f6fa06d56dab519cbae33c`
- fixture hash: `sha256-38158baa488957f4efebe2494068936f86320ad50d0d4566b804a6468d20bab5`
- score hash: `sha256-61859e0fd12f17096401e2a6c0a0ecad5417adfd66005bfed75ebed5406d4175`
- bundle hash: `sha256-d3cd30701bb664426b8547a95529882c10472889d56031b2db3e8bf4ad984fad`

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-03484d17a35834258fb4c116ac777404bb054c45515bca8f4bdd31e9fc0dfda7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-42ba8ac4e3a11b1b1c069a018cb222936be7d4c22ce4cc3e73975cf521cdce13 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-933e5d4de12f46ceefd9d8df4f6eeac0a42e218c75ebdb511d38a912ff3cfa63 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d12573b6 | sha256-cd9f8f53615261c4f6ccdff30db2f41073d95d7a0e86960a77148d0f10d608cd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d12573b6 | sha256-de7067101f27bf3128793eba865253cedc027fee7e94cd6134938306f562c0ef |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-79d505cd | sha256-8879da7b298fe38c1b078a599a02c73d1a62fe3ba3baa4d73f8de8305f311875 |
