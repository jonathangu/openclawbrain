# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ebe19ca9bc459ccc52f05cb3ef8e24277b70f060cd7939f534aee99c63488ae5`
- fixture hash: `sha256-83e33e2dc3d5736fb8b475959b3f799a1522431a9ceb8bc4c7fc74edb18967c0`
- score hash: `sha256-0b71948e8332f6f5e3f9687eb24ebe6f9352f11dc121a12b4f413b4073c06d2a`
- bundle hash: `sha256-a4bfc5588ef016f0e6f85d6aa1457a13ce272a1bba2ed8a9aef16ea8249bccc7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38dbfef9dbd8d3a664a1f95db7f92b5a52579781d0eb6bee8aa47758b54b5ce0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-733e6d07914ee646968c935eb2f6243f88d47cf3ab82d3988d586405bd643a91 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94dc05f893853c8a8043aa57de853af96acb8dcc7584609883c2fbf8f3f36a36 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-22aa1f78dcc9dacb126690c55472dbefda2c6e688a5afb79298d8d82f880ef35 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cc192b9f | sha256-06d2fb6d8c290efb16773ddd994af0a4988d301e43f85d44654305fcf65c53b7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cc192b9f | sha256-ae59f22a5b10c5d771bcd447230e844b9da33f6a5b0319038864f0cb71786cd3 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-f654d062 | sha256-52990b1f30be7ffdce65862db8f44e6137dcb0a8120a41f1f89264f812ce2173 |
