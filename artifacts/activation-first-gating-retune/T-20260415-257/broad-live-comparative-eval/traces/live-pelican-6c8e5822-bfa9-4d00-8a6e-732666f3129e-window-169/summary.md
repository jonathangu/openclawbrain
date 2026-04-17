# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fa0a3e5a2be78a517ccfe2e1e4b8f4e2529d6e5ac6ae4838bd2c1da5073ae788`
- fixture hash: `sha256-9c907c31d6df545ad3189fb66d2746fb0938842a92e6704858a51c0bdbc6d6a3`
- score hash: `sha256-aaa1a26f6794a8b064b9c8527e1195775a6bdde8b06e476a6ec5cc9a645305dc`
- bundle hash: `sha256-bf2f77ff7b4ac8b228b7bb933fad612103d033067c6ca8b303b08608cfb25581`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-984a47e062d84a4c2db4727f0e783a355cfd91d65c98b2c3a27a24fa9103cec7 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-926b2c28f827cc02d99d8c2e0007718edf2bd6bf96eca3b6c54000e3e0e3c72f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2076d9a303a082f77a6574589dfdf574c0062089d88a6f0e0c54b3a8fa6425f5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c8afbd204f223bda6cd52408b518a88d700f37a1a10371c67d499559cd06658a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9a47700a | sha256-4b153242a12beb96a14790cb140a7f35b76c814224a4f629ae93b78f09842785 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9a47700a | sha256-33a77fdf01eaf30a9c406878dcf7576e541196f8ca7fd0588040d2731a5f4c00 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8d05ad77 | sha256-b92111542f048746201dc7afc43943cc9e463b1a5a7cb3f7cad8803a908e5137 |
