# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1df02420998a45ff18b6fa7592e1d6cd553e69e00670f629819f48f156232f3b`
- fixture hash: `sha256-d26e41f8ffbf777f72220318ad80ec7f532c81cc4e8c86beb0f89befd769d272`
- score hash: `sha256-e816ad8952b9e0e7768605f519d5ac42264084305fdff2182a6d11207dac96ae`
- bundle hash: `sha256-ebf4ee5930b2800c2239c51c6371fcdd8efa52d75458a95551bacf73c0454c36`

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
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-d9ccadc3eda4f12bf710188051a463abbacb3b335d3ae23815b1001a8d14166f |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-7eae22e6202e946265c9f7f97f632669d1772dbdad25cddf1c8fe5c5feb855a8 |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-2e2e8b0d19772fee4eef2bd1223889644730e1e28ece1d458a86b613c2252050 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-86015ffc | sha256-aa4647d0f8f69d8d1f1e914c10115e343d9fa153a8de1c03bb1928570ec2c394 |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-86015ffc | sha256-58d2bf22aeafa63f9286f8c09c30a4c113b935cc0756693f0da7afd0e10255b0 |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-c5bc7c07 | sha256-a281d22e30fe673091f8e4f4ff98c59e840a72870e183489a94f30865e9a65cc |
