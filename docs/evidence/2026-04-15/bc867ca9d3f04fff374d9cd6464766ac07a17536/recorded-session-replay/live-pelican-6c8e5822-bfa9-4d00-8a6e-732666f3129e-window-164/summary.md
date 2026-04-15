# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f337daaeed6bc47fc68765c0195f530bb4ce38ec076e00ac4c73412b426d85da`
- fixture hash: `sha256-5e5cab708ce5b294bac69d34a6279b47e648ad8d40ed85f35998caca6e589c7b`
- score hash: `sha256-04a383631cca557ee8c10d7ddf8753ea3d676a0ce27417a8d05ee2725b177625`
- bundle hash: `sha256-40762f8040cd1be62ead7416436cc40b0b6fda57104bebd54bbed01cb29c3ff8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-100c819b269094add6922ee0aca0d157fd41366c476c3703f8d276f1431d3315 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-47a2316cfa4138ed2569ebbbcae648980b3600d83a233fd8d3084b92687570ce |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3e9c8b16790e3b420d315f3ee6fe8cf1c7aa3480ac3001a0c069273568b3a1e0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-0464efd3fec008b13cf77176b09e4f0e49f299daec484435de8c1bd1fb4e0a0a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-901037c4 | sha256-85a7b61fa3c09749587f3020b52243bcd761682d1737532022bd7db2401157c4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-901037c4 | sha256-aae8483b2d031d65e46bafe92c0545f2cd80e766bf37f230e89c46ec9f7cabf3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-901037c4 | sha256-85a7b61fa3c09749587f3020b52243bcd761682d1737532022bd7db2401157c4 |
