# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c80a3decbe06cbf3c4af187d8a5af847ce341540f23d409b6e7d63d31df4bcc4`
- fixture hash: `sha256-741cbfbe2c3d2f3a4ab8e97bf7b8405a7d1cec581f3191dded735c7802b1e00f`
- score hash: `sha256-5390743af60108cbc028a123b9fdb31117ed08a995042217afd9a7604d07e20e`
- bundle hash: `sha256-ba975c73d01fa45668677e3bf660dd1eab6e927f6c74ed7159d99a2a0d135be9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16c0b7f4b283cf0cadf9518aed3354f26372dc3c9867fbbccefe14e243137800 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0ee73dbbbd7b3e09fa07ba074d161356322b0b1eaed303c360b0c04c7b70fb07 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-12f3d4052d18a3f7865e553c2c13326e1a4f769d09d05917a4ee8394d9cfa8b2 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4cf0e27e55eb262b67cff99fb1789fbf5de2ee9b515fb2dc29c6bea255b5a34b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-daf6710b | sha256-3625daa77ade41666e3396c80d0e1b8590af9a61d8dcb311c78b31cc10491dee |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-daf6710b | sha256-9a5c18e0955f77bc17dcf538a2cd0e511a096051959e879ed3229ecc5955ac6f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-efcad05c | sha256-ee0e3da468736d24b0a2c0de38bae7a3bfc17425c77bfb28a34284c9bba5c17f |
