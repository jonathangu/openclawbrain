# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ec7f0ca39d2ce8e4aa075852c984c14df45efbf7ebb099adc4d8318c646741f9`
- fixture hash: `sha256-1eeb0e4e14f003831776523471001891e5f51483edf8cd0fe82b3b2a7a4e72c2`
- score hash: `sha256-671dc17abb675be197b0b3e3fbd51bd861c8a65d6e5897a73c6b8685d90dc7ee`
- bundle hash: `sha256-81b30a85e542ae81638dfd6f5a1d7cfae4a52e7fb8ef671e71ab4c3fca959216`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d39232e9e4182be91b475d1dc774e142ceab1f9213fd98395428e4f29aee341f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bc58f070aef5e055fab0bc896d9508c1fc6aa873521fb31a824562f77dddb51a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f0a8ddc6927810bb45a91133e2ffad0069df281f741fbfb3b2283db5e76228d8 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ec519928a8c5092c28330ea01c8217184737a4e72caa882c9859872833f27c64 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b97d2b20 | sha256-b119a1cf0e8901eff91be766de9e6fd616d75a8ae608b973846607a8e2e98dbc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b97d2b20 | sha256-a0eef2810919a0653c5617803d7f74dafb2cde5a9740ba99d1a500c80b8ce318 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b97d2b20 | sha256-b119a1cf0e8901eff91be766de9e6fd616d75a8ae608b973846607a8e2e98dbc |
