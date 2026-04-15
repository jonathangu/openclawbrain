# Recorded Session Replay Proof Bundle

- trace id: `live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-e0e56ffd1c26d20085e7a9eb3248f58dfab8c43d92d6bc35e804da203ef4f7d9`
- fixture hash: `sha256-e4b8d39277cb985d3e9ee559f9e373775182720bfc10b6d9350141f9c5016460`
- score hash: `sha256-d6ae1ea78c5bdebe89aad24b95a95ad7256b33ae41d1e18d7f5c14e1b06039a2`
- bundle hash: `sha256-8deb42ddf98f3298bc6e826c64e0734f4136602500a3e38ded45961037bec43d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 80 |
| 2 | vector_only | 80 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 4/12
- phrase hit rate: 0.333333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0bdf6c0bfdc77dfb35df2ddd80b080b8e6bbd2f8f1020fedbea4770e769e1c72 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-10c236fa923a3252b60db0b8595c68b919ef14b72a23419cbbae8123f9a4e3f2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d1019d9f90076cf78d96541e68038c0eb4593d44d294380c1d5e0cf020478e06 |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-1e5aeb74efa8817a3bfb63376b8280b0d47b92f64d3d0b5ce3c9b23c4f95a9b3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-c26d71a4 | sha256-d60693b3543b6acac084e9c5a2cb02bb2d26e51de8f08cc39dd09a56519c4c13 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c26d71a4 | sha256-38cc7c981eefc341930f58676d88378e3a983d5e24c44462e3fe514c74e88990 |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-c26d71a4 | sha256-d60693b3543b6acac084e9c5a2cb02bb2d26e51de8f08cc39dd09a56519c4c13 |
