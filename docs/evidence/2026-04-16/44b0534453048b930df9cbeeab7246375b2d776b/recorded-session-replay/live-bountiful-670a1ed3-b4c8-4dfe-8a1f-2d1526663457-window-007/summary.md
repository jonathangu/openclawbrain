# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-829093e68c8b369222680a9ce88928380b9f7729d760e59ece8cf8d1e776b82a`
- fixture hash: `sha256-31ba926396eebecc30aa75781e7d614cd75f3d45744f5fc68d2426d0829db138`
- score hash: `sha256-a057eeb69ee0c429eeda815ffa5b3ecc08cdb7dda70d02a6044086d0acc43f81`
- bundle hash: `sha256-fd63879e0bbf98fdecc7a67bc3292d4833a378ef26e9b74d4d8d893a51ec633e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8d38776b0b580dc292e4970ed98776136ff0d2acc01ecbb7a8d527a0c51a84c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8bd28f260c1c78e18c9c2efc9fa0baedaf4b5e61237f60bc31fc61154f0e2733 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3242de8d3c475246b27054707464084c6a3198963cd3579b04cf1b386d1a67fc |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b922e9e992d218df6ec59599b6a15b6bb5d49d71b6a68a5818129e7624194d42 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ca97dc57 | sha256-dbd26e9c00e8ea1ec6dad4b2733cc6051465bafa4dc5e6e813a65a32f0112ba6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ca97dc57 | sha256-b1ef7a1202d05cbc56ad82592f3c5730691101bccdcd502c35fcc04f4b657cd2 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6cc6f84a | sha256-b97ae9f421a03a70b37622d7bba4dbdc041adc704b6d989a101d791a679cc10a |
