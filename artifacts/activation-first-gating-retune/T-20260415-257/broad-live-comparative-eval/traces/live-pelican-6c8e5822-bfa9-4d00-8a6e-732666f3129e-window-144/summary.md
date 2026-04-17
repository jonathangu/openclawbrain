# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0149e3caf93b3375cf02c24f74af73ff26b7bc10ea672fea0331d56ac334a82f`
- fixture hash: `sha256-8eef5aa851168050667187c6a1f16965243d4107da455697233fb94b6cd8be15`
- score hash: `sha256-7aef2ef1775ce7b3e28e4c1c58b7fe0dce1b092d015dd5d1dc47bb58f41da9eb`
- bundle hash: `sha256-85faeae2ded917a69b0985b5bc15d059530b536372dce53b13fee6fc38a34ea1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-beac0c48f82ed7e8a11f136719a9c12038db11daf2070f49f0ee8d4c618e927a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-45c74d531dcc100850b93a0ad93af83eaf50274589ca276935259fbc4a3c1a11 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fb3cc4abae88c32e6be202e12233ef2c087cc122084ae3f53a1ccec983a67231 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1c7c39ea995ba7995d2d453d9368cc84da2f32b6c0fa43973e79567edc8ffe8f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4e08af49 | sha256-304966ef9216040524047d662e93f4a2e2124dde31726f8cce61df7211dd3f11 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4e08af49 | sha256-1afd60f3568178a2b5bbc762beb454dfde907ae6fc8ac201f567ad25e91a95f4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-882b12d6 | sha256-e7b42df079f09ae23dfccb845382b7e2279e4caf0e48f891b3ae8b77a4b05a96 |
