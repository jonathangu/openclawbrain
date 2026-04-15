# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67782e30fe5f9982125f26c2ecd77317f6b86c34b8443a476ff968e4172fc9ad`
- fixture hash: `sha256-3275c723fd5e55770c99a0a3826bd67e0749405b630c9523de493fe0719c674f`
- score hash: `sha256-2716219e513071311fdee3ac388fa2824b0eecef6b46dc18dc83694515a20e36`
- bundle hash: `sha256-e7ec4656390f72fecaed9e8204ff036e371424cb475220d2a9a9fdca8ea9f0e8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6faa741f27c297696cddf75c51e07e62f9d376795b5d33f012fd6c625e199a2d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b9f5579a55637e638054344462db08710c03dd6760d9fcc89f8e9223c460a573 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9407c3803b2333eb7b8e4a434154127d0ae3e5e85059333631043150bc56dcab |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-30a63be1aa0c82d03170a4cc1992f317165631e8cb5d10624c546b6f5ef928b3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cd96887a | sha256-b4a4ecb876655a4e8b2dde0d1913fe542cc93239e27d0e0b7e193afbe2635d4a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cd96887a | sha256-b4a4ecb876655a4e8b2dde0d1913fe542cc93239e27d0e0b7e193afbe2635d4a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cd96887a | sha256-b4a4ecb876655a4e8b2dde0d1913fe542cc93239e27d0e0b7e193afbe2635d4a |
