# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-804025b59e9ccd56b6c987bbd4ac39904cd2a5df99a64be4a350dc631576367d`
- fixture hash: `sha256-e197071d3e574f8d14c2a018d3a6d553f258f6b2a618a40d8d6d516e0727c08e`
- score hash: `sha256-e2edd49023740990eca198ca593bc571459f22b54b9cf6839444078ad577bd5e`
- bundle hash: `sha256-5aa1c688e4b138ca3c0846800e8fd4c697d219ff869e1d64bfc357688f765868`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9879dc85fee5e21c2aa8d0f905f0d82e912ca593acccb3a82008612064aa877f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3b39da1cbddf72943d957e80c707196a98aa2c3392cd7552fad8d075ade4c94f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-47d89c0b1e2b1ef21ca1987de4476dde114580bf601415f6ae28a27330e430fa |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-9aab74a71413513d07ae2e08dffb96d06d55ae711df7557e60fffa191f9dcad2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa279678 | sha256-6e91d41953cc1a0f8a31dd5fcabf583f072f76601046f10576e2805114d8d096 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa279678 | sha256-0ba772442c843c78cddd1e74a727b643375fa25772bad5999cbff640cb3a8c71 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-aa279678 | sha256-6e91d41953cc1a0f8a31dd5fcabf583f072f76601046f10576e2805114d8d096 |
