# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c80a3decbe06cbf3c4af187d8a5af847ce341540f23d409b6e7d63d31df4bcc4`
- fixture hash: `sha256-741cbfbe2c3d2f3a4ab8e97bf7b8405a7d1cec581f3191dded735c7802b1e00f`
- score hash: `sha256-0b0272acc964e1f3380bbaec5d40add90f84cb11a981256594b4e3f239167f2f`
- bundle hash: `sha256-8fa3a6be794ffe8344b15ea8ddbddb7a96aa5283db3fe6ef35be0c38e065a57b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16c0b7f4b283cf0cadf9518aed3354f26372dc3c9867fbbccefe14e243137800 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-734d4c856b7858eea77351a3d45a3f54ad6ff786aa4993caac87ca5534f86e19 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c2a882aafce2e3b0f514a2f79d37b404d1451a56334d3cf6a887bcbdbb4e5633 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-74ac16d7d83fa3e17a1f38093101505de5b59b3dc85a47122073193f2e47b44f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f9209299 | sha256-d7ac21e3d33e43c1ae4ab09f48a28c3c49ae27ec5c124163b4baa2b80c6a4ffa |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f9209299 | sha256-a2bcedec0d6e48519ab4fcba689b3028ed688fcc631b3c609290fe5a1124fe0c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0df4f1ea | sha256-f0eb3d7a7cc5257f6b7245874311b8ed1dc51c8e14f67a9a078c4abfae3200ac |
