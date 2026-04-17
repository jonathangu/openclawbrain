# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f579c26265d9087760a95275a1ed5d3c29a7fa2a5745f0cd5985ac21a42da923`
- fixture hash: `sha256-344c0f8fa42bcaf494090e8fb4c4629c475783bea29bc527602ae9b6d23e9791`
- score hash: `sha256-c31eb0eb6a26106dcdf13154ac07137a05be4cc51c8e9905edab93f251e42aec`
- bundle hash: `sha256-093dd0993ed45385e78a6afa75da62be49b7567e5d0f87e0d2b41a3fcfac6c41`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c8dbb3069efa791ffae92b155399c68bb15a7550a719f9b3772c99bdfa5fdc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-631ec1d58b4c85370fcf428612763a4055356f765ea154d369a5f89c87782505 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-823d138069d1945a61befa9d4a2c3835bfe2d576f91f509c1321ed371608206d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-272147cb7fe47fce56a6bcc4cdb5344d1db83658733d3d175366a2a15b9d9b27 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-14b0cd0d | sha256-58c7b6da036907ab2977f98e359fe2683f37f96516fd68003c7ab8b750322410 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-14b0cd0d | sha256-4c0b5d44be59911c6d74c173edbff1b6410b85426f7c90bf1cff214e22cb9625 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-57b6295c | sha256-391cb875df3752c6b094cd3c5e8421b0a3bece1421a8f6efad97b10f49cbda7d |
