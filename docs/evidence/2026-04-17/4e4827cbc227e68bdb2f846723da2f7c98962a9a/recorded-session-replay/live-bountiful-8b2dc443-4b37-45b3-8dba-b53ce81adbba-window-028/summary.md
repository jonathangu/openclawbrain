# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e5cecfcda3863d55354a9b67074a3c6ce69c277ae2b3137a3f72bd0dc80700f`
- fixture hash: `sha256-6958fe867e36da1beab1df863be77bc3ca8278fa4e3d5aeb7c88307e08cb7f39`
- score hash: `sha256-db94f5fa997a1e307531e72035aefe9cab6d963e16271541e0e8fb22f843f86f`
- bundle hash: `sha256-245cf1891b730e4d9787e0287b93d46f21fa99983e3c38abdff40b3165f79778`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-150e11cf8d8c4499ff46b946af855e658a817920409432754a5872047686e4e4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8c25a9bb4eaffc70cfeb167ef848d91e541986ecece364c06645c28429d00231 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d54759ecfb639217295b315da3afb19f976377ddc486ea605b8cda29389afb02 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-46c36079 | sha256-aaa17b541d2d55f9f11101729e4d1fc3fb4d9d958063c864ca1f57d332f1b8ea |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-46c36079 | sha256-8f4db017167991da85caf9022f8e1ba75c3c60b4315e79671a7fa21d9a94a3e9 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-3ca3099c | sha256-cbdc5ab7962bcdf7fa7f52b25b2bebe24067d511629ce377900446a310ab734c |
