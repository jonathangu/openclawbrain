# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7095a4d9ce26969c4dde9c329e749be730ceb1f708c47df4f4c59a5abea7434f`
- fixture hash: `sha256-107b047d2badf45fec45fded8a1234ee55c336b1a2803fdeba6955f2f30cad1f`
- score hash: `sha256-5ca8d019336b5194156ce466a3d72a6dceb1437eb8f7c4ea137681aebec26177`
- bundle hash: `sha256-9aab3a4f139efe4c2518a79b134296579cadccf79281054a6093c84f4b67a648`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8413d38761902f8b7b6bde87782ba48c8aa416069cad02d85c57f922d6bd4f24 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-62d61f32b6a05740c74daa3f7ec6476046643b864f08e514527301b05d5fa75e |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-391974ff649abf60c1e436681111f9d283c1db9d465b7b6aaaed237f18254c04 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-58e911291456f2c45d90df3614d3ff22b2a5ceb4bd5e36a1f720b3c710de7dfe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-aa6ce102 | sha256-524689a3c569ef101d0a000b7cbd5221ada54f47b6d2a762fe9b63ccd7888075 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-aa6ce102 | sha256-5b25ed4878a931c68a9288e10c99f726cd9f0078a28cfc4eeba456e8c43a9bc3 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-aa6ce102 | sha256-524689a3c569ef101d0a000b7cbd5221ada54f47b6d2a762fe9b63ccd7888075 |
