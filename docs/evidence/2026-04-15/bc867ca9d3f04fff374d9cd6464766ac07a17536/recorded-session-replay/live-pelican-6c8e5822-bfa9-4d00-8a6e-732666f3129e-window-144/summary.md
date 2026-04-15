# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0149e3caf93b3375cf02c24f74af73ff26b7bc10ea672fea0331d56ac334a82f`
- fixture hash: `sha256-8eef5aa851168050667187c6a1f16965243d4107da455697233fb94b6cd8be15`
- score hash: `sha256-fd86c5d5ff0148c0b9e711593b36a7fa78cf64807aaca1af8965e52c0855a314`
- bundle hash: `sha256-6d61dd5def9544204f63311f3ef7ba622cd1a155925575b194d05878429a6ae0`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fb301310aafcfe8b35cb0e5cdda027613ab1f4b213b5578c6f398f82e2130758 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c46f23bc26697becf251ca782ffa5d49e90d7a76e38381bd769accc6050f2d1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2369f6699106756ed6623a59ba4937f77fbd07e47a207e10c99b78c6c67900ce |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d6a05cca | sha256-e6fb16ee6f6a4f1b7d45daa87917c4a2c43b8c13ecfd5bdb234a00cd239aeb1c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d6a05cca | sha256-e6fb16ee6f6a4f1b7d45daa87917c4a2c43b8c13ecfd5bdb234a00cd239aeb1c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d6a05cca | sha256-e6fb16ee6f6a4f1b7d45daa87917c4a2c43b8c13ecfd5bdb234a00cd239aeb1c |
