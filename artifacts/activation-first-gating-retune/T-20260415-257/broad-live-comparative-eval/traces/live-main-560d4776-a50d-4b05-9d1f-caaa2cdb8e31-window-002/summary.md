# Recorded Session Replay Proof Bundle

- trace id: `live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-035806e6f9bcb3753f58456b70d56dc8a01f4abf60114aeaf359384806f6c24b`
- fixture hash: `sha256-74bbbcc2ba3e23b87dadc56cde438b46daa30c3743245ccd0b40d24de1249370`
- score hash: `sha256-e22a13f0a5ad0b1d8500bca7b7a699df8b944e132b77a3dd4ad58dea1cb0fcf7`
- bundle hash: `sha256-d103763cd8697745c416cb8ee9f8e32b4e2d113ee5c5a4f557a69dd90b356908`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | vector_only | 80 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 4/12
- phrase hit rate: 0.333333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54ce8361598e1b4080ba115badc91e906441ece2076bc91bc1f9f28df2706034 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-ef1c03444c5ae7a47251845d70d3d6ff40775d8ef8d0627f7fd1130f290d31f3 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-bc8692577f06dbc169a0a3fe2a978fa5d019bdb08c47a6d83eedc9308d2fa4c3 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-24e6f1efee442df4c819174109b5bfbf4abd604f210f8bbbf60a98c4e6405b50 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-b36d2171 | sha256-cbeaee15ea4d5d79896c9ded4d7f9415de09e429fbc66b959fa1f428fa485a13 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-b36d2171 | sha256-030e9d4bf8bafef5a7b5b63e7fbca93cec94ca8e480d45e8e43dfbfa2106e0ee |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-612dfa90 | sha256-52fb95f371bec44dd3a0faa54035296d0b5959f4099e5f02dbc66d74beca7472 |
