# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a2e597e92fe22d4f55c094b8ed54b6a9af6fa4591283d89702e798da892600a7`
- fixture hash: `sha256-622fac4fb2f464038d17b973948a3daa701456585a35960e995213dcda72d3b1`
- score hash: `sha256-9aa9ec3263a28fe18fd84bd514478d142944bf790cb2498a2539306ae96fb532`
- bundle hash: `sha256-42ace474128351a530cac08f61949c590cd4f7b459fc7c5e6471b532164df078`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-937c7f8b3de6cb0ba567e2def00dcbf253af96c301f9c26d07a7c1aa6375230e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f688a567027e1f588485da28dc2558b19e069a2a044a64e6f07425b9ae848417 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dfc82b58bc3ce821077d3ee5ec58cdd475b54330a962af79ceed7fe82cc86c0b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-1620e7cc333bf94740b7e77ae9a831d673bfa9d11fa7f65d9369eff4453483e8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cab91405 | sha256-d5fefaf428fba0e9b37a2fc14cf444adfec857fc703a95b14575bda46a746378 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cab91405 | sha256-14bd997144270cde131cd6066e6574d8d5ad7895cc3ed59b761b1626f8ad073d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8ed4f5ce | sha256-e4a792e2289559860fec56f4ac6c7ad03db6e3221e808e307edffac708991b9d |
