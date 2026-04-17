# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c110d52fa9d814d5415fbbb31a6466d9c241d27a71e256584d4f8da38b7870b3`
- fixture hash: `sha256-7aea2d0a1eb139bffa0a7ec4a62af3e2b3a4882d28c7223058abf7f69edb1954`
- score hash: `sha256-ae1914bdc4b91016a4e7bd1a32d27f0b94ccdf94d865387555f0f3a001fbd2f5`
- bundle hash: `sha256-9ca30448af79850914ffffe1821e4f4f885534b490d798d2b6a194965a09c41f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3c530b06d54c1f577e737bdbbc2ed643a0051ad5929f4a0256034473b3d96cb |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eae207948c51ae26fb04b083ade8ad77c7fb7c0c7232be4cdcb8275e6d2d934a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d0fdf104897fe0e04da8c109433261ffb0d01a088498fd36d457ac3d5fc52460 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a70a8c0b0fef56a8cec90008a5c0021fe8635aacb76e449a885b9bdc91f0eb7b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1515cf2c | sha256-688e84c5e1f91110831c709a614b53f9c7b52200e870bf2073a0cb727f504f79 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1515cf2c | sha256-b4ad86135632a074f2be1ef3de6f3e2d13c7e64515133bdc7bb5f3a992280540 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3f6d6ea5 | sha256-a9023940d90da57637a01104ddde27595a54fc7a08d4ef40a08b08d9a859d66f |
