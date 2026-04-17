# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304b5ee53cc148670256892da800bf0d31f07b699447be9e8eaaeff5a3c2cab5`
- fixture hash: `sha256-60dc2f86ac1ee754f931ba95c5a33382b613c3b1b0a7e2c96deb303d2eccd093`
- score hash: `sha256-16ce557c2c23e0bd13f437b6578c84a9fe4920dccce8fe15719acf13b322fe42`
- bundle hash: `sha256-adb8c2713644aefa36de1d1ce3cdf8ce07701a76ea87b02ecb907f5b7d0c6c73`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81d1a7582801981771b9bc27a32c83725b8a8a67e2715cd65f17099531df2d18 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d6877648cdc2b4ad1e4fc503354f4d027869628a8607e4f809d3fe55929270c5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-267b967f0865101babc7656045989fa7cac4a3182b9af29b43b8268cb82ebc97 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-dda6651f3353b496a155572506fbf13d9ef6d883c577921e66675060bc32848d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-73897f66 | sha256-c3ecc44f1719946e6c87e2640b4b16f18e66610676b9935b34331cc19283dd3f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-73897f66 | sha256-528acd5f72d65eabca71bbba2e45a98c4fc6bfbd5be96df6d259498eeb2b9c07 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ed58581f | sha256-878e54d3aa6ab26426d52d06734d2a218a5ae781b094384bd73dc65e24ea085a |
