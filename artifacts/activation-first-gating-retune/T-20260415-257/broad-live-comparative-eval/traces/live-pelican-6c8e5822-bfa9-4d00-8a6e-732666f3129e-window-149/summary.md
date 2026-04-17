# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149`
- winner mode: `graph_prior_only`
- trace hash: `sha256-299806353ab465c5dc0556cb46d4c0ddab82caef7c74016e1f229b80f14988f5`
- fixture hash: `sha256-6a01dd18700c95a1ef47fa69bd96f40af05494c628d07e9816bb8fa24129ae15`
- score hash: `sha256-4f3f7ff79f5437d69e0473ae480e0ba4f7d1a0eb9083af0f3ee59638ed4ca852`
- bundle hash: `sha256-43af858a142badee25c55561c292b4e3a11385153360d51ec13c2587a1e0b4ed`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fd2b5dc7e86f33e7f35222bc8995c5714891af6e48c0b188589cfd85f30ab7cd |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b52dbc3132ce822a9ab401236bb82316730dbebd185fcad62c442cf4ca15ae6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c0b1524285f99b5a77b99037ec7b796956509e972ebe4b71ddaaff43e02356af |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5bf13516a1f2316a3081bab397a905ec1eca8cf803eaf825e55c565dd4831b13 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-86d8ee8b | sha256-ed8594959f70fc8b33322e60573d96d7ae5a9920e66e0afbf6aad7bdad68d1d5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-86d8ee8b | sha256-683c5f405597453172dc08977479c45e9f633d68cb956f8117f39b1d8e1ee520 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-2cb5edf8 | sha256-b2abbfdd6f54dd9f51fe24a7413468e34461d3fcf86057af3b73ded1441b4ae8 |
