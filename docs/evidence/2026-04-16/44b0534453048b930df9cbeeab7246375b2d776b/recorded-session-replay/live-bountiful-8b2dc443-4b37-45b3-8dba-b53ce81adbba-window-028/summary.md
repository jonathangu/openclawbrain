# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e5cecfcda3863d55354a9b67074a3c6ce69c277ae2b3137a3f72bd0dc80700f`
- fixture hash: `sha256-6958fe867e36da1beab1df863be77bc3ca8278fa4e3d5aeb7c88307e08cb7f39`
- score hash: `sha256-cb41c71185ba4e7412189648087518bd72cd53c7289f47ff9be8401b8a790a5a`
- bundle hash: `sha256-f2ec24ae9cfcea5cbf240c6f2160fd93ecaf691c96185190b0ca24dce2e15662`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7381932aea4d1bd30c10ae36d19326006a8cb4cb3b6e5b2b2ae6dadf03b6d135 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8561d2aab98865a70c9b662c729919db8e61875b15f8cc98f9e958bdeed6ad13 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-935763a5db9472e505630cbf134eb7db3fc5c1283a9ce1cda7e1577bf4039b55 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ed50277651a189d1e8f7041e492fce9879de648a9cf4fc6046112b7a4f6e9711 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d14073f9 | sha256-1e33b9eed258c0ee815ac56d14b55b330807801e7dcad6ede3182caf83441e07 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d14073f9 | sha256-39cee501c06a054ab870452c3214376dc4862d91eedfc9501fa9bcb8907e6228 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-c7201d1c | sha256-622ab4f40cecc383b3418e266eec8d676ee1f3f7d73b9146975f1bde2acc1610 |
