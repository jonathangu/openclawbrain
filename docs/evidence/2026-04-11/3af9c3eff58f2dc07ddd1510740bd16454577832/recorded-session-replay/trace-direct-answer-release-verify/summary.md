# Recorded Session Replay Proof Bundle

- trace id: `trace-direct-answer-release-verify`
- winner mode: `graph_prior_only`
- trace hash: `sha256-01b84ada8d6846964137ff080684902a4e4cc0e43a8cefb78c04b2a1e32acc17`
- fixture hash: `sha256-aa503d9d77de4a0a7cad9548aae31476d1a1a5a96b73a44dedefa1b9a484712b`
- score hash: `sha256-2b79d5f8a8fe889b8d548f79ae169559744fd0b095061af143b5a6abb31b9800`
- bundle hash: `sha256-bac0e0dc9ddcca6cbf71e260d700e3f52aa46b31680e80dba11a6037e8c8063c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 12/16
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/4 | 0 | 0 | 2 | 1 | 0 | sha256-50e4253f1347265eaa9ff02792d8234c7d8ec16a6a093d8a6a69ca92ec831ad5 |
| vector_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 1 | 0 | sha256-47908f9b98d73ebcafb6dd0c23905ad6e0da3aa7f65b19054b73ca236c5491d4 |
| graph_prior_only | 2 | 2 | 4/4 | 0 | 0 | 2 | 1 | 0 | sha256-41655239d6cf380ca463e3e4a8499b09a1d9d76699bdad6c21d56759ed902b92 |
| learned_route | 2 | 2 | 4/4 | 1 | 1 | 2 | 1 | 0 | sha256-2a02fcd45ccd39bd32ed814a9741c36fcc7197402653686cd7234fe7a779b6e9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | release-verify-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | release-verify-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | release-verify-turn-1 | 100 | yes | 2/2 | no | no | pack-fa349f51 | sha256-b1e646f249bcc76f88ac009eb1d35e92378d7453490c0037ac1aa986d1d93129 |
| vector_only | release-verify-turn-2 | 100 | yes | 2/2 | no | no | pack-fa349f51 | sha256-dc9ed5a9b5e69db27e16eae74c1fc8187e1c6637c18fb735975e408f2c4675ff |
| graph_prior_only | release-verify-turn-1 | 100 | yes | 2/2 | no | no | pack-fa349f51 | sha256-b1e646f249bcc76f88ac009eb1d35e92378d7453490c0037ac1aa986d1d93129 |
| graph_prior_only | release-verify-turn-2 | 100 | yes | 2/2 | no | no | pack-fa349f51 | sha256-dc9ed5a9b5e69db27e16eae74c1fc8187e1c6637c18fb735975e408f2c4675ff |
| learned_route | release-verify-turn-1 | 100 | yes | 2/2 | no | yes | pack-fa349f51 | sha256-b1e646f249bcc76f88ac009eb1d35e92378d7453490c0037ac1aa986d1d93129 |
| learned_route | release-verify-turn-2 | 100 | yes | 2/2 | yes | no | pack-900f2c71 | sha256-885bed9da2d736fbf9fa4cd5b0ea3521b71da4602052c6074818773b44c58b6a |
