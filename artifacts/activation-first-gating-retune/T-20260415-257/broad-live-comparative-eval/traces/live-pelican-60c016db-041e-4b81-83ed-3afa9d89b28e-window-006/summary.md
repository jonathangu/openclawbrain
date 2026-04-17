# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15f74481afa0ad3c49942a752d93fa21610759dcd0f5184c05ee667b747607b5`
- fixture hash: `sha256-27569dfe07b6cf66e357fc072347afe0c073b0dd225ff6f7f6dbd4f6b53bd5c5`
- score hash: `sha256-dfd58ad43a5948bac2d39575179f8e907acc805128a445195bf3f1c898621316`
- bundle hash: `sha256-61b4e4cfd50f4949e66017d622236d0055df1ee2d6d427340d6c2cd609aef170`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d9780bcb02b4dddac9cfba41582ad72477a9d4e9b030a1ad3ced919c347c5d08 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b270e234e56aa515dffa0f343e005acec4c862b6c3f5dd291a3d8af65fc07dfe |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-63c46413ee9778ca6b8f849bcca85501a48750e0b609a1b682b1532f5cc51b5f |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-20bcbaad13fbd647646d51486664a0a22daded57b840ba5a73c33a94ada633f2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-132c2881 | sha256-49cf4bd9da215ce9ba8aff18d3091220751285c422c10f1c7552fcd50ae8f491 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-132c2881 | sha256-222e94a1e78c72673e28114b1ba30c41c13b6c51421e17632a38b2648fd300b6 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-26d2a4e0 | sha256-21cc3ce1dd6a868c4ce229de2e6b56e17bc475ce19e2b0827010c4fa06aaa4f7 |
