# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-838b9295d0df32bf17309a7744670eaab3129f24a6dca2ca9110c4b4940f8ca0`
- fixture hash: `sha256-56f7d90cfb38f59327532bc9b6beae4801650c72b03cf0a3e492173ea24b06f6`
- score hash: `sha256-558ef9b3474b649b5c7f63f2392d0ce346d917863602c1d1214697029f6d1862`
- bundle hash: `sha256-2c8ae17ed6d958c1f6101bf8a5166b3dc6489dc95a3a4c6400a3ed2515438581`

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
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d12703af710851e5a23d60b1d20c78b1a6044ead7e09a16f607df5e76e23db43 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-3a81a12b35b5b3db628ab8859eecdd1639e0fc8082d54d6521b846cb68d90514 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8e9609d8fdb1e60d773b98cdf5dc47ca651b576f592b063d452c06af07cef4d |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-bd5a09c9649ba0e6c8c3ce9e1d44340848dabbcbc173668bbc2b3bbfe98b5202 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-14501a85 | sha256-bf5130d6db448d5d0b9112ffa6ba0f573c0b6fa1253221e7e31bd6977b206a42 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-14501a85 | sha256-e9ea3d1aeba20f131f6832a43272cb8e5d4cc754cae5547b13f6f1a19241f2e4 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-7f939090 | sha256-f85c51c00a2d2e019576b06d9d1b59b0774645e899cb440919348f3ea40ecd6f |
