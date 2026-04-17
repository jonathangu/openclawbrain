# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147`
- winner mode: `graph_prior_only`
- trace hash: `sha256-12b53203712e88b756dee356041b3ddb0e18e328e1c8f8ade691064553599eca`
- fixture hash: `sha256-8ac6a4fe3950f0ed5cfb2e1b9bd9c7ad4d79faf9e22bb913250d8fa59920cf2e`
- score hash: `sha256-1fbd6eb4943e0912754f0b9b3de4ec3d93d36489aa66b56e6f6ddc9c4ed64398`
- bundle hash: `sha256-d8ed614d7c37e15a36d0a7342f8a7bcf25df3b66155e2bad9a1a4b87c0f275fd`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-55766afad53c9e202670418bdf755c0f71228a26fa5f954c36b74006ec3fe092 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e9c45b52e08f2ed1a7bea9585204af1f0d9272ea530e6996c1c46eb793c624b3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7d13aa8ca9201af3ed74b35ab50686c6a9f87d40d4c81adcfa98edf1531b25a2 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ece1cacee073873f90a178fb211c4f8c863d0098661be08d1d08e1bc6dc46391 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f95a541e | sha256-ec2ff3e1815df30bec7071aab5c908d6ff7d86efbf2fdce1125a1143a9343175 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f95a541e | sha256-c5b26a5a23c908678f7b96c64266806af562f3ba211e1eeb0f287e21748d4cc4 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-fd073c5b | sha256-0978778139ce26540bd2e5fd47f65fc7121b49679283324836c1bc98a0366f7b |
