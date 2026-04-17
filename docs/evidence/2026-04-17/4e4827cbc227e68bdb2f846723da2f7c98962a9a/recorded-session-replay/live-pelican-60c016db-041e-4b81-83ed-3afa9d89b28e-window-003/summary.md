# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-954e7370fea342c102d12700848d801f9cb4f766c26a152991092af096940dab`
- fixture hash: `sha256-71ae1b688e319cb6bd41b60a17bd7289e838d68bd59d5831fb46d3db379a64f9`
- score hash: `sha256-2d9cb567cb21a8b0f3e931ce4de0a8c42f9171836b999acc0c247f97d13fd437`
- bundle hash: `sha256-942cee35d50c5c8440907b41199e422d040f1c3dc3a6ba2ee2b58706e0173210`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c508d0d953d2e20d8464c0689a6d8d9a5c0442d5d485367bc8ffc01689888e09 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-276b2e73f26d60faa069fcb0ac298ce5376bdb39118cdec2efddc1a513251c81 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6ec27eb368d4003fc418e11f148fd76c5738fc3f1272d929aa5fe30863a2d873 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-42e86c6d75755dee95c7452f0d587f8ff735c87fc04f1cb0dfd0fe5dae19740d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d59f6f7d | sha256-37fb320420c54e34e1d9396a70b8542fe56924192b7a1f0bcb54a90a7ef226dd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d59f6f7d | sha256-2e16b7b6180b333d840abd7122d5f3f0c7e29ff223be5d2933f223be6e32b91e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-357c290a | sha256-4c42511f4cfc82c3fb066282c8d377fb647d5107654ae8bff8758d3694dfd71d |
