# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9c0b90297f99ade602878feaa8cfde6e3a19db0e47440bfe22629154903dab61`
- fixture hash: `sha256-1baf21d3d9b73bfb53336d6a81b7f65e4d6e7e9fb603fe4e8af018eaeb0d47ef`
- score hash: `sha256-0c5616a2c216777d5db99f4caf5d44d4b765022fff07e2d1a652614a2bbc0eb7`
- bundle hash: `sha256-75a5f88859cb44da73eb5833bcff1df26a960946955708ca0194e90cf94f5a08`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-486788a02d0e87d1d309cf97a88d618ae0b9c6a09ed3968593c77e567fc45c0f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c8fae4e51ccc60c9348913b1c4295f9566e3f1140f5366fb06e7c9719267bf2a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-67484390e27c20de93b4d1551201c2e3f1969eafe2bdb3dfb3dc27b3ff2f24c5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-517bea70 | sha256-450eee23c702f23b1d7d5cf5cc5795a234b6007c909be7d03cf4e0f7d971e305 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-517bea70 | sha256-450eee23c702f23b1d7d5cf5cc5795a234b6007c909be7d03cf4e0f7d971e305 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-1aa603c3 | sha256-640c62ac93114832a44e5dd0d0349c987d1972118a34afce79a8d0767db9a83d |
