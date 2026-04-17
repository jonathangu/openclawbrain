# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f0b0e6f922517c200d53829cb727fb37d8945bfa0f48cb619647397c75b1c77`
- fixture hash: `sha256-41f6d7ac9ee841cd833f6dd48ee4c826e9ee5964cecb194b203679cdfe3cc453`
- score hash: `sha256-6ed43ab4ce1a7f413d031afd1d6ca2070764779ba536bb6ddbbdda96f3458c2e`
- bundle hash: `sha256-d040ad32cfa70543464c9177741e406b33622f775437df9b56a21a89ed6de96f`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3be8b80f288ad1443027bbe7441fea408977a916cc24a4d025e0ce74fb942938 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-45d6362f504df9a99115ad2493b4c8fbc3d482539cdd0d420aab3d1fa9f3d53c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c9d124462d1afba576c477d9c33285e8a26f59f97df40e4463077e618c503647 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-affd92567ad5c9b43423529041fdb38146ad22a8bef8236fdb3568b779d9968f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4c21d0b2 | sha256-59144a5eb851c4ebcff519335f52f6753b9d1baeba5f045850421726070df596 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4c21d0b2 | sha256-758e343c0a24f30825cac9f4a4199791c2f7fe57c0d13aaf8ea39aedec63dd4d |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-1175250f | sha256-c4559a0900bb93080b2cb52bc1b5c8a7155429587c7768cee9249792d3c4516e |
