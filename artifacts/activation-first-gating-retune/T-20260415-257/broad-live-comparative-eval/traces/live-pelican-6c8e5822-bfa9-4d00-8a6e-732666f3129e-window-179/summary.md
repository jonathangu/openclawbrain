# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eebf5a156737b8a1b13583833520fd34225ae0f30b4afb05ce10671f54ea2108`
- fixture hash: `sha256-63d1e3dd69e143127b58a78b17f85b8f588fdddd25950ce30e59877032c4d44a`
- score hash: `sha256-29568cc2719bf022f9fe4beea9ab3b8fe77b3c4c73ca4a5ad437843106103f2b`
- bundle hash: `sha256-858ad6d6bda43b5440ffaffd7af6f97047186efb1a9c3dccbf83d592cf00be08`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-798f7833fec1b062f8a2789c97b9a979ee4a90e5e78bf32289929e17459a82fb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-35aa915194cbe4b3e678373fcbd795144bcf184b213cd2de4cb4b92c4c8d65c6 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4a978f8bd6c43b6704b2eec581be2904126188dc9bccda3535a7d503cb3afce0 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-10bf404f13575351ac4a9a401defac9dc10b4e8540870aec6089f6099d4cb46a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d8e8201e | sha256-b789a1987ab626875d0dff1f60a1de7efd9d1470dd657d2266bb36635b14d2ba |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d8e8201e | sha256-c5479451e8b60dc7c10400a89ccd93fef5355c33f7a1aedecbed44a47f418804 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-7f16ce53 | sha256-7a10f59095c6e52d78f2624ea0dc44c4299758c3ce0ae6ae3a48ecf45a9b7780 |
