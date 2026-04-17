# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f4905564adc9cb953b8b5504309a4080c3ac583fe0f629cb62b1e05f91ea23a3`
- fixture hash: `sha256-0ad0b5e1e0f2271069ee0d118e38a8f083b22de4d11f9b10cb9ee63b3ed54883`
- score hash: `sha256-49948eb5856ef3fe50468ed72111fac7bda88a1e7a3f25c1ee3839f434e8a005`
- bundle hash: `sha256-939183cd6213a5a0f4392e3ee11ba3257c6badf0027035be3975bcce0a20254c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-64badee520388e2e251dcf80ba87d74776085beb63219f4be30791f06cfae40c |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cfd27ce797bf519132193ef876475acae45b4f7422704ae528c12e390e8a390b |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a000aabc87e62f7a1a0daa2cf139208aa91b30de54f6726d20159ca90127ec2b |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c2a65c8c4403741ed562346ccea9f1a34119bdebcf8295e2b39abbb213d8f6be |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a2436640 | sha256-b71bb67a894103ac792aeb976215f8201cbe9bed617d38d1160b0b83cec86a13 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a2436640 | sha256-904e056a34c0ffee2ae6818e551e4fb046f068dd3009c18368a400baab24f2aa |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-69bfbfcd | sha256-ba802461abfb7fcbc1592c58ee3d647690106f7ed0ef117319b2405e6cfd3639 |
