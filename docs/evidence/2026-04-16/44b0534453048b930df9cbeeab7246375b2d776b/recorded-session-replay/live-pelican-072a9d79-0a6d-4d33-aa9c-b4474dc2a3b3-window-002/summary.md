# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1df02420998a45ff18b6fa7592e1d6cd553e69e00670f629819f48f156232f3b`
- fixture hash: `sha256-d26e41f8ffbf777f72220318ad80ec7f532c81cc4e8c86beb0f89befd769d272`
- score hash: `sha256-f7eb00a0a0edba1fb1788e80ceebcf79c226ae85b29d816c4eff5aac6148a6df`
- bundle hash: `sha256-e3fd3ffdd746cfca04cd3a1b70cc9adc978668d1299eb07ac9b9d23ff0837863`

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
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-caffae64068969fa7e1d950417642498125ccd7a52b99fe5538a0a0e555ac8a8 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-f8c8ea74850d946728be06f9e2f0e8d887a087612ad788771e958bf00b19433f |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-49a247a5593e052c9146f1058c869316502742ece10ec74f834ab5bf687d4809 |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-dac8b61b2b713b3e572ab83d0a4b4097c9a5035a9bad0522a31ee72d66c593f7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-61ecd68d | sha256-cfff04bc19da53e5fc92920fb70d14e80e8a2382220e2fae5da8ff20d1bb9f6c |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-61ecd68d | sha256-d6a97a3a933bc9148c9ad9bb873ae95a9c2dd6bc8248cef1434ed534813f15b5 |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-a1a7f298 | sha256-6bb700118ed8eeedc05cfca3f31c302b80d501c4fd18e1de7ef4c400d686ae82 |
