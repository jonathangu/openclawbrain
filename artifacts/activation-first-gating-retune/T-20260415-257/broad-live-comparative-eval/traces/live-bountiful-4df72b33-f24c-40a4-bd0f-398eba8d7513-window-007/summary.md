# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b0963b5e18b40bd244437e7e0701feca11058e18371c7c7830706c53a28f15f0`
- fixture hash: `sha256-ad1fe96e9866fc7227d860e828f71679e46d996ff90526f19b5279748e32ad9b`
- score hash: `sha256-e1ad0428241826520d23126b6fe51a42b33d57e6862b1fd11cdf034acbb5ef7b`
- bundle hash: `sha256-39d9487e1846355b5c69195bf543af9871580ed973c88f5798ff5ba6d05d09cf`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-63768ebf632f25773108234ec4f8850307fca437412854c0aa69b01f97c1eac8 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-8e4360cf26cbe3dcc0e9a1c242df88a69fabdc43dfb88fa708f1dfaf10a9d9e5 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-3c83b1cca22b526e229944cea56d88632257d56fa74a739ec68ec515bc265d8f |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-cd7ed4034d9d17b3e8a9895c669489028b1a1d966e4e6941c76dd2a63ca33d6a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-b156e850 | sha256-2491d5dd789b2d0a741b461a900aab78d0b06d599fda4f4a94ad1a20d74b95c6 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-b156e850 | sha256-c91375bd9ecf7e83b88f24f9b1edd932b11481f06962ef50c2ec4a64e85a819b |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-62b331b7 | sha256-8ada7cf2d9e758e7a6756c8baa073c96f5c2d36f01719ec4a9462718d4a16f6c |
