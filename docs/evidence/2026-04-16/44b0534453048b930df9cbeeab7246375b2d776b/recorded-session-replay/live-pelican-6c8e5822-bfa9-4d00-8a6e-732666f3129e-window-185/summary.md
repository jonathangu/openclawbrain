# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24e1e9ea471d19d207e35c598683b69d84119849186e1c11e6ddd97932c4aba2`
- fixture hash: `sha256-11642bd40eb6fe8c9d53921bdb1bbcbbdf6e5f35f00a6469f30893bcfb466a96`
- score hash: `sha256-310bb3437a87a2bdcb81886d2b7c5aef50cabebab8360d2ddbf4edca5946a8fa`
- bundle hash: `sha256-4326ea59339e3114a68288e54ffa6b79aabd200ad26acc486c52acd84c7fda47`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe10a677dfb68dc498fb14e838ed3e08e036ad9f9df81513ada323fcaad39838 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7d3a6743f3123b1e381a0f3ae070f1581a894bc81bb15800e7ffc870e703012a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-01717f6f01684db23243318b18582b10d347701b0e785e45382f3eaf92305a73 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ff8359a9fe07ebf811aa5a3f9d4276e666e5f9be18dae54b15f3e1390490d9c7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ae1a763e | sha256-aabd008770325e2ee66b7ac48334e75033b573c6e389d74d6312807a6e40c111 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ae1a763e | sha256-1506fb3cf253fc3f69225fadc1999558a73d20a75c71ea1b16ae61ea9725b8bd |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d77b9001 | sha256-87fbf1f0fcb86b774255e2505f4a5b572419063a843a228b1203e50b3317cfd6 |
