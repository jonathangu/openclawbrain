# Recorded Session Replay Proof Bundle

- trace id: `live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4d327d918fbcc4a369abe7ef4d164f9a5cfa04faf7adb2505c432935e6de9ae6`
- fixture hash: `sha256-6153c43221a8c0bd8b8f42dc9046e70b2d1a03e5bc667d5e9fc62b4aa1f0fcb9`
- score hash: `sha256-6605c6ff4315f8a734591494649790a17019f1c62388b6cd45c32dcfc56867ba`
- bundle hash: `sha256-f1970589f5106a21722e1e1ff5ca9de7f85902e91220643fb75b63156d63a2e9`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d542a8f0c800204dee4f72d85787e8cb1b923c865594f3befa573eb5cd2d9388 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-340e46e972118a3a8cb6fa4c4ccede34bc4f9dabfe0b1c474875aa094338c32e |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-30dde90d4976bb9897e533e672418c20dabab2d5fc7e074df958e0b49b9b1ba7 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-3ad32f562a981e70abba288424a87168524c0add35a94b3e47d036da0b889d50 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2b53868f | sha256-c2cf56542d785dd60e286ffd57f58465801d3da8e525588296d3196d677f7316 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2b53868f | sha256-73dec33b0437c980847bbe78a002ce5bcbc06b7d592c26546aab79f7a510be3d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-62c37340 | sha256-8347754bb07f7d18186163505a14d9facf7ac594c12333d95424147b008fb44d |
