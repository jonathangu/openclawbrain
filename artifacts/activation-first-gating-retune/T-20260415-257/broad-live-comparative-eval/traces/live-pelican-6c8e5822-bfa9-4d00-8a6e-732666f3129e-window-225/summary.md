# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7768c82b82cfc9e79b11d2862950b229d848ebccb180ebcb1860140cf56b1f18`
- fixture hash: `sha256-e3d4346656fea9fcd52a8093d89ccf43c79e719fec02594aace8851b57c7f190`
- score hash: `sha256-43ad12b6d256a0c77950bd806a6c2440c84be8605355c2353f50fa48f4acd2f1`
- bundle hash: `sha256-56033634402740b03135ab693756a43ae6b8e6ca66e917d8947e8b22752bebda`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb4473107ca7b170cb7198e9a132dfc26d383b8a4567d404be160b76d2d08390 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-22aacec36413db638527dc7a522ec997b975bb53a0d2d5722465b22f459c246b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3f4289dc9f6ee317e3315f5e54d31a332c32571f5a4f003e17fd03bd10658430 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-03c815bb4c99ae74a355372704cb5c49dbac8065b579a2bc8cf3e5bee5eb140a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0e022cb9 | sha256-9b0e45d19c6882f5f8093212a7ce07bfd81eab34ef4279a19cb2fb96c521d1c7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0e022cb9 | sha256-380f72685854328538505b155a3f7a64130b81e5fe0833ee2a01839aece810fb |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-87fef1ac | sha256-eb551e2634d6b1a4a06fc94e524baa6f0dc06ae514bbba47e5ebd9f077845941 |
