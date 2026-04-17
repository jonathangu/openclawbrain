# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200`
- winner mode: `graph_prior_only`
- trace hash: `sha256-438b689c90e1516f117c130a44f955ebe5121f19131ef3c8af4f3b72e782a392`
- fixture hash: `sha256-fef64d4e61173927de1b8c7e42759f7ee5918ab3e67738573626a046f39d5b5e`
- score hash: `sha256-deb7978e569001b734ca8760ef73c3f1634941aff5cd2eb22b533c0b4a91c6a1`
- bundle hash: `sha256-b66c1ccb3890314db5fe0f68895e9a19e2279c6fc479b8b6e31a1fc5d886b458`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f3e6c6ba4308832d244620436e1eb71e4969051bd02e8a257e4c9a12dea8653e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-98301215bb2496eb2c9c4bb46697727f3a5f15456ea450a23b5baa8e8a9a605c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-efb81e4bf227f510fcbe0a809ca4d18916fc5f0c6ae07d2c1e54092de60ba470 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2c27ac431dda26d5fafd1bb2b77e0c97b2a50345ef2e16ce31e81c79d3a601ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e03435cc | sha256-b86aebeb126445479f62d264c838f66912fd94f1b67f316b5b5e8161b85fb788 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e03435cc | sha256-18e5dd4e4cab4427eabce47d2db45676c93fe321293d82cf94ef0075131327a0 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4a51bc13 | sha256-c9584a3b76d48fb1b2a972826e24eac008217c0d132eec8435d0f7884dd92307 |
