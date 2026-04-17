# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1daacbdd680cf4033ed5d9fa2efa105e6544ffe1129c7600ff85b76c0c2f8393`
- fixture hash: `sha256-ef7e749ef838de36d236aa29e0590a88a86c6b42be12cb84bf00123ad9c263a6`
- score hash: `sha256-abafd0dc3ea80840a19aa126c0af1869a142064377e8aaee239ff37aed20d510`
- bundle hash: `sha256-79e2063f10dd55d9cbf3247d3626a280a4a022d97760f383fa5baae38a32065d`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-715391ab03706266e4dd92a9d6ff099345f003fb7379bf779cf731b9d18a7950 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-40d1745fdaf88a8256e9c6207dddad9f5bf3b4d3252e71b947d88b0f4504c332 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7a82678ab859c186232a8b65090885a9f65600862d4af3d71550ec206b39112d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6648010f8e12af6ad2947c11260534f9b0fea414e9391c2e5d7f720fa8727d32 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc631fd0 | sha256-a476c12b1ef9e5f3bdbff240b03224512aec0ce428e8aec68c7b8c823047014e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc631fd0 | sha256-a832f3398c38766c38ba138346f7be53339f74f8f4a52abb48a211a8b10d8d25 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ddbf27f3 | sha256-fc4fc816b16d0c8c5ce2a5608b750498d89905dc53f0dfe948393bfa0da377be |
