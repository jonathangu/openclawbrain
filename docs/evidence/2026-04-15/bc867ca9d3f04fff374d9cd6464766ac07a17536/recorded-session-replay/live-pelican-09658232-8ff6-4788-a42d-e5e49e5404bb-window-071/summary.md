# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071`
- winner mode: `graph_prior_only`
- trace hash: `sha256-aa6afe07711fbc8a13484cd14e70ac82c78cc503ee5449452a36b775fa63c3d1`
- fixture hash: `sha256-bacff39860081979b6852dc7223e7e30d3e6e8700496899a8864e78cf3c36fa0`
- score hash: `sha256-43858d5fbe548aaf01e29e62e5c983a6c36926bde2367799de6e1e7bf8a7bfc6`
- bundle hash: `sha256-c023b13057180104534435b535a4661cb57e566180d47f9ed3fddfe240425210`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-080c0791d3c8d4b27935c18a06ca48413df84ee848ffe0bfd6099d007a81a298 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f5d1d621b5e2275fac2c23a0160a0b323c0eb7b4825b26d7d172cc1132954151 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f4d06d769a77df6101170d3f11ac9dcc750db7de026edce7f0fe5813141a5890 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-ccf7331b8fd61dc2df3d9c942fab04a340e9adfc98bbe446810354204186e257 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5d541cb6 | sha256-966ac64deea8c0e1f78a953cf2196516e0e1497043efd5230f039eadc76b34a6 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5d541cb6 | sha256-6b72bddd516df32ac8228b9d33db94bd924da4bfaa57a6e58513f289d286b535 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-5d541cb6 | sha256-966ac64deea8c0e1f78a953cf2196516e0e1497043efd5230f039eadc76b34a6 |
