# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f9684c38d91e55a983d42052df21e03bec407bc3f34393946fcda8e1b2d39f4`
- fixture hash: `sha256-5b6e1bbde60f4bcca2052f19249d943d07695521da1e7e8b46846e97b143bb5b`
- score hash: `sha256-faa8b83cd22eb0a9f359ef04108bb90b54981f82aa4021f017e6802473fbb1b0`
- bundle hash: `sha256-cb596245dddd5bef5c8418e867c5a66ba02c61677f9f944825138001946130ac`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-74a0ec494cc9b5ec66bff70ca0bc3e9262d5754f8a93ce7d222367da206ee232 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-86266187c9fc718a78ceab0b6244ff08c3a7deefa8ec554ed0fd86bfe6423981 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c827be3fa513dda09d480c685ebb959526a2308054caca32ddaeb9e97e9700ca |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-aefebdf7f988afdde9598afa265e51c71baa8d651a84af124320b9fefc6f9211 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-03558823 | sha256-a85a1731fb88ac268f10af9c2702a98823c64f9c8fadef28a82671dc746f4ed8 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-03558823 | sha256-0d652d50a2fc6233940a4d14c64d07fc9da59d2ea99dcd09b3ce94e2c11d0f77 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-03558823 | sha256-a85a1731fb88ac268f10af9c2702a98823c64f9c8fadef28a82671dc746f4ed8 |
