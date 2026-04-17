# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c03db6636813d15ab314e1036640845c998fa263e7aac887b43bd18e611eb255`
- fixture hash: `sha256-a5250b921f3f7515a2c7a2c53cad821f0baf9317cf0101574260207ef30568c4`
- score hash: `sha256-5e1fab06eca3a09aafb8b7093f5513c8c59909632e697a28a4a343edf296f74d`
- bundle hash: `sha256-d9174ef739d22d5faa33aa892e60cbbc4a63c76a6125c93f4478268e235e396f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fda1d2fc29e8cba170ae68633dbee693f7e6102e2f256c7ffe0a2134e709581a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d9e42141d43b2db965178eeb1d5ec814216c101553b005234201e15849b79be6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a18fb58f18663f2b42a1ebacfcdbd63dbb4b40186e3e61f8dc69e8f12262e6c8 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-efdc2fc547ea8751467046439c4d326a765a6c50427437dbc458356196091714 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-94b67fdc | sha256-9faa171d330a2f1c53607c4dbbbc5447f8b7db6021d2137935e62e3462648ec0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-94b67fdc | sha256-a2ad5624179107cdf944686f727e82a42fa8759c93e33cec14a5d2db3f55313a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-4c921c2d | sha256-4b3a8b573a61a146dca3d874f5e7155a70951df9b162d0024881309527522318 |
