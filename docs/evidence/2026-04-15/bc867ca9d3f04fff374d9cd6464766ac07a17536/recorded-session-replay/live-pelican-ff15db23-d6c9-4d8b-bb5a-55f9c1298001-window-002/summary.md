# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9abdc2f8435606514daaaad4927f60e901a9d2b092eb5d39df77887ebe5a304`
- fixture hash: `sha256-6e857ab9cb3ba1ec3e0f72cceabb24485f23daf6db41d61af726b2888aeb0f66`
- score hash: `sha256-30b442edeaff0873f32f38461d787e4cb6f2a7665e67b30cad7217637dc027f5`
- bundle hash: `sha256-5e7b19751f336d1c8910dc439bbe6af0d700e8b97fc5bf607d4f8a55ae13e9f1`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-865378fe979515e6fb05b86bb93e571f4e3d4c4ed17ab843485b9830a42b2636 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-51da5d1deacdd35f9a94f6e4d310ac45cadb93cb0e486ec23424f36955eb7d78 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-991de3207300bbcc2c57d779a12fea8987b5ce91b4f956cfcac45a28f38726cd |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-8a653ab765709fe0050bc42f524dd3a4400a9177aabbb187023b263e73024356 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a2730122 | sha256-9558ba8f3aaac6f4629d52f792d9476225d7a7b0ce7e1707853ecea4b7a5ed8f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a2730122 | sha256-d35c25f83008679c6400ab6c88d4746d466e5659cfd278b085ed8485f18fbd1b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-a2730122 | sha256-9558ba8f3aaac6f4629d52f792d9476225d7a7b0ce7e1707853ecea4b7a5ed8f |
