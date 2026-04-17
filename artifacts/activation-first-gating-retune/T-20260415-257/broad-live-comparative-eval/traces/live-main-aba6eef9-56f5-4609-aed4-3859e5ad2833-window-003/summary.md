# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7206817fbe9864fa741e2aac4263783623734861273e6c92294a7e71e4bda31f`
- fixture hash: `sha256-3fcf85ac262f6dca9a6b48603643e7ca5bbe3663229b7fc7238b9b7fb3303591`
- score hash: `sha256-2c235197497e2e4b42b53bfb96e79627b58afcc6aeba445aa304b0a08c8efa91`
- bundle hash: `sha256-ef4122cb1ab366601f6bf5e81938ac1b9205b8ea937b757440f6ba7600a85490`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a6acf0ea4807b1384f37283996c98fb6c5d3cf32e52bbe94b1a201a85fdc539 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5887ff9e0b2f898cf0bd11970832996dc8f1c493af520ba2e46560bc4e19425b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-59b5cad539cf6c4c16b752b95af308b1584fc68abc36b420e86df2d5c52bb142 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e983ef850cf45687c3685a278ddfe6816ac0427e7645223c82ff5c58e1b3a92e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cfa763c4 | sha256-00a391468ea1b4c1410b5670f9a176cbf7bb9372579ab6174fa059bfd80ba24c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cfa763c4 | sha256-ad811a14a8ed19a7f0ddce2de5febc21539ddd52568f83ef98215517a38936ca |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-f43c6e3d | sha256-4c7dbcc8115182618b475855c0739a3e56997b1b4232be18d1a7c92e666c8ad2 |
