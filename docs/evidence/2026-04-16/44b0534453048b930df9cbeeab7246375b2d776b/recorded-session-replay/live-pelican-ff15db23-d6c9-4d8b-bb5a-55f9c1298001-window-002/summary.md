# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9abdc2f8435606514daaaad4927f60e901a9d2b092eb5d39df77887ebe5a304`
- fixture hash: `sha256-6e857ab9cb3ba1ec3e0f72cceabb24485f23daf6db41d61af726b2888aeb0f66`
- score hash: `sha256-aa5a7a656bc35a2bf28529b9d0b70a9d109bd0a2dec0e380088231500cdcf598`
- bundle hash: `sha256-88265a9945a334c312394b2f2d5d1bd744551702bdebcc421811c77f772a9afc`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-865378fe979515e6fb05b86bb93e571f4e3d4c4ed17ab843485b9830a42b2636 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-16d9b78a95f50f7692b546594f19950d44d30e3d8c180fbb5a89b61dbbcf8628 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-055d1c1cce94b890e165c15238294413d917aabb5b8a26188e0b37fce696119f |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-4c76741bf7d7cbdc18c48a967f044f443fd837b9cb6ab1802f527487cf78569f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-48a5b940 | sha256-389fe9dd7c1bd783f37fcc11cc77ee5a43d9dad2f5228a830d34d43e3513b5aa |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-48a5b940 | sha256-a844e1121149f717c898c836e10f32305def14cfaae6b2b0072b690f51ca8f57 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-d8deceaf | sha256-46782ae574bb5818677ef0f41c396209d8b02c27925cf7e7476f9c3028deed4e |
