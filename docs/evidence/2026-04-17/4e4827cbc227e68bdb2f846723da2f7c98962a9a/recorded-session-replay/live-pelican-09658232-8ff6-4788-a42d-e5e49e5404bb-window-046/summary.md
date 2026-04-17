# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3c2d2ba443dbef189a04f697781c21859dae784757f070f6d624a5c22c1fd87e`
- fixture hash: `sha256-d5090234178376a892e6c521a05dfe5104bf688b9e6c7c68cfaf8797d0e0e324`
- score hash: `sha256-06d3d6078f7554c27f2f3ff46b6765e317807aae52cddccc444fa2d8b687b027`
- bundle hash: `sha256-e4408a344d1c41ac0a4f5613d03a7a3fd856164bd70a55779a018edc9b8003a4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30cb0b5562af7757589ca1411482395b52af039eced1208652e3e0610a2b0728 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1cea7aaf76a13cf489e134b6de2c4e37664db0cec14a608f977764bc3de6e66e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-740dca57f7ef3bee200dfee05f45c2cdf1b4b662ead20cf023ab2e01ebe25b82 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6ac2986e720ecfe063e75010f2cd65e26566039b568aaf91c95dcb9db6d2a5c0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-09789d70 | sha256-9e1e738011d1796a292a2c3d612e4bf00da9bf91a5f068adaeb66c603370b126 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-09789d70 | sha256-0190eafc60347aa253d69cc6b23e1456c43cf502870d177cfe73b64497ce643c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-fb695a3b | sha256-b6bce7bb2333c3d36b6d321c68e20e786a5563928c5d9c128e231273639dd2f5 |
