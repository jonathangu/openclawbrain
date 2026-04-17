# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-838b9295d0df32bf17309a7744670eaab3129f24a6dca2ca9110c4b4940f8ca0`
- fixture hash: `sha256-56f7d90cfb38f59327532bc9b6beae4801650c72b03cf0a3e492173ea24b06f6`
- score hash: `sha256-36088425b6e7ab79c0b46addb4f0367f16b55e398942a16ec64e1b0e04932ed1`
- bundle hash: `sha256-1bf5238e9b6416ce043372a3f11ba5a0d627990e4dc6d897cb9dbef8a1f0aebc`

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4feee15e592dfd0c9866546639284fce9c4c80de2651038dc74933cd5945d15b |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-142ddd6656e00ef346abda95bf58160ad18480296eb6feb729718e170f9492eb |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-829297d025b4f64ae6f940e950024e5b4368da613883d7bde0bb12df45d129fc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e16ca4b6 | sha256-1d4ecd484c44031475540e301ab3fe2bd6424e12257e1082e41a21d03a713f43 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e16ca4b6 | sha256-7ab67a165eff7d95b7022f94aabd96c66cdcba9564990c73bccffbb421593fc9 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-4cb01ac1 | sha256-914fa108ba73ccf0c6f6f2a805c04e4a1dad25941547e706a56132172d863070 |
