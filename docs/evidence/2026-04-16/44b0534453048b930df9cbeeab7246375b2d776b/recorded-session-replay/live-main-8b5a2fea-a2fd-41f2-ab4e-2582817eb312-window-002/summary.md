# Recorded Session Replay Proof Bundle

- trace id: `live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-e0e56ffd1c26d20085e7a9eb3248f58dfab8c43d92d6bc35e804da203ef4f7d9`
- fixture hash: `sha256-e4b8d39277cb985d3e9ee559f9e373775182720bfc10b6d9350141f9c5016460`
- score hash: `sha256-0850fcdc9f0ffa6eebe129b7c2f33e018f86f1613c69bcfd978f48c909de30c3`
- bundle hash: `sha256-8e257e72d1dc7bc456ccf6c39631b1a4fab940ff71f226f47435cab1b017f9bb`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 80 |
| 2 | vector_only | 80 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 4/12
- phrase hit rate: 0.333333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0bdf6c0bfdc77dfb35df2ddd80b080b8e6bbd2f8f1020fedbea4770e769e1c72 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-6cbdff7dd2beeaa1a2cd41cb7261991dd72dd3e0b4315924ba354cc079b80832 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-88c2f4d4e0b37493bb66c5324a32fa4e68f8678f7f0c25c7758f7ad8fa123f53 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-4ee58cb73952c31faf06dd3cf2feb1b5d35bc8dcee8091b80fa5ffdfe43929b5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-59332952 | sha256-3b2ef0b6101616d02e41d083f5905ae126d756c53628dc58410d51ccd2813f87 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-59332952 | sha256-237b428d9506226de5989174d4e56bf0bef45e7c3150c2152eed72fc6cbf9f82 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-bf5df00b | sha256-91bb79d70ce5a898222190509a69598581969253c64346c9443631d47544cb7e |
