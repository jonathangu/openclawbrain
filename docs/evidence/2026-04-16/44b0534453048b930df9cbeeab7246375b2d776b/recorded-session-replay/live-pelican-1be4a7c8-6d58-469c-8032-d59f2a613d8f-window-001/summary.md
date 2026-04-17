# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9c0b90297f99ade602878feaa8cfde6e3a19db0e47440bfe22629154903dab61`
- fixture hash: `sha256-1baf21d3d9b73bfb53336d6a81b7f65e4d6e7e9fb603fe4e8af018eaeb0d47ef`
- score hash: `sha256-a83454a60bfb1cd590fa8f81bc1f5b9ba880f2928e6305b3a12bfe34c87ba1e1`
- bundle hash: `sha256-8820dd0c7fb4ed41a7f0260e84db589c87b71622527926155a7eeea4fc54b5a1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-27b2a69331fb76743637a0a59a8c052316c43dae2eb924cfbe90678912704fb5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-97a62bd5b9cfa00ad61355131c1de790ee09deaaac0b91cd01bf7fbd9e1b6a1e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0fe7ba725528390a0b1423d06fb7632c83320c1393143fcad4fbe76be20f4d28 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ac99ca3f952016a442c0f4df4b6168fe502fa2a3a7f924776234fc7a8da0343a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-240881eb | sha256-f7a5baec0f9360af593f71bc38f3a2ece20179e2d3d1c613526d28edf3e09468 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-240881eb | sha256-f7a5baec0f9360af593f71bc38f3a2ece20179e2d3d1c613526d28edf3e09468 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ed329b3e | sha256-284d0d9e7016947fa65557e9355f62a065af7efd35d961fa99b827093736cd9b |
