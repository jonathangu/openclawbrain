# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-681f87e7efb93a7fe2c0ca8693ed89cbd98e19a9b1cd66b4274d0a49023027a7`
- fixture hash: `sha256-a3933e3ff4510f68b788c54f4766a4413b4c8e5f41767b34e144aa18224f9ea0`
- score hash: `sha256-edb7744c0f9b27e776a1d084c5331e248ea56dbad27557dfb28a8eddc78c5941`
- bundle hash: `sha256-c4e8a9476fda72881531709000dd704b3503aac7bc75700c4950e879240727cc`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3fa9eac509428c290df212041800a61f9388237d673568a94abb436cef2cd1a |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-892a9d37633002e867ed1137a029c4006b2dd1bb23aaa6cdadd15ec71416c259 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-971c3dd818b6b8a7f6131912cee03bf951d0828264af2c883bca32a25cd97bb8 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-3cde3aac55f6ce063346fd5a3ae0803497721c1b8fd32b003a76d32252aa01e1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-a99539c1 | sha256-3d7a8136cf00ad65bc45a7aba7b5803c78de2f40f05c671ccde74f95b7be11f2 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-a99539c1 | sha256-8742f0f72f2980f1927f4c6f305b25e2aa306f4223087d14dde845750474f841 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-a99539c1 | sha256-3d7a8136cf00ad65bc45a7aba7b5803c78de2f40f05c671ccde74f95b7be11f2 |
