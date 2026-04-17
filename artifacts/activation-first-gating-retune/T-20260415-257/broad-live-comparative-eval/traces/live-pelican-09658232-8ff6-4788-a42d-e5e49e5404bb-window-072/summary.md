# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ea52ae43f846ec5905bcf93ba8cbe911779b358315c7c925dfcb3bb8d88c42d`
- fixture hash: `sha256-5ca126c8a28da22685d19c74b8dc7e5cb0bac37c0b916d2162c68f83275f6394`
- score hash: `sha256-bf4c8fa7bfdcbb319b8bafc6a8abe9c1f2ebe7c9f9fb2eb7de20c077a7a0b4c7`
- bundle hash: `sha256-cf2e1b8c5087d3bf1f2bf7a60688e78b6a73c3c8ff2a3ae06647b278790e21dc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2ee8e31cf8824d425cda16aa09e9361fc2028a17b7c4fcbcc21c2fa64f147edf |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3c8731262e5d2d3733ef0319103e6f00561c3cec5de435af5985d1372d5c9e41 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9823069282ff19dfdf262904b57a6a39bed09209006fa0c489b403fb22e120b4 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-a69680427d3c9f081de96eefa0a91fcf33c556e8e8a72d70a3c4e8af52f4a9bd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5169fe28 | sha256-89e3d6b1e5be0ad96dcbd943ab98d3c9f2937da8ed4c5f549811fed2dae534d3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5169fe28 | sha256-175e40d16c4509ee71333c6bbb0a876bbd1fdc6fb1ffcbad3d3140d0d01aff1b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-813a31fd | sha256-eb8c14953be9e58d3233122689338ce88393c122cef2e04e433f6f934640e087 |
