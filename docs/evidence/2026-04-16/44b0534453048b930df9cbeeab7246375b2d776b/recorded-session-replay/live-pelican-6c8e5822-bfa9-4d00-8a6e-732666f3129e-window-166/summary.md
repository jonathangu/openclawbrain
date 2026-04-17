# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ec7f0ca39d2ce8e4aa075852c984c14df45efbf7ebb099adc4d8318c646741f9`
- fixture hash: `sha256-1eeb0e4e14f003831776523471001891e5f51483edf8cd0fe82b3b2a7a4e72c2`
- score hash: `sha256-511a7b41684f374e0f96970b72f4cf3ecc5e709331ba138847fa6b0ab80c4265`
- bundle hash: `sha256-0a73555c5feb21fc43828963c4391d5c04e41f24a489298da44230dbf3754d5a`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d39232e9e4182be91b475d1dc774e142ceab1f9213fd98395428e4f29aee341f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-340aa11ba6f5ce96c3df6a3888e4decdd78515c522c42dbd3911d372b10af916 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4bc03a00c28f85c15f8a4fa9aa98f0cf5a260541ddce62d4d6a5665cce637141 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-795fd97b9748ffeba6e9dd495f19d4bf7851ff3a4cb8165814b60b71bbf68fff |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d6a003e8 | sha256-8ca08c9f93fcea5680062debe0de3a1a7d1cbba0674168bf3cee9f408b6ea855 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d6a003e8 | sha256-f293dc35738674f0739e4074b745808cfb0407747947223e6fc6afcd1a4bf4e8 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-799aed75 | sha256-5606ea246147951021d66440e01949860df7869fa263c0e6378c1f71149754a7 |
