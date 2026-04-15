# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051`
- winner mode: `graph_prior_only`
- trace hash: `sha256-00bf2bd686f7cfc027d3b7749683ef5ae4ebe1a8b4b5f12763771285b87ec8ab`
- fixture hash: `sha256-1287af06cb4b83146712d66b78f07ce6e6ac74450d156f3cf86e05b95cfe0f1f`
- score hash: `sha256-85d30dde8f3757c35d5ddb6468ca2cda0dee21cf58621c2ef81d6001f8d1f98e`
- bundle hash: `sha256-4342aa2c769b6aa11c25123b1c1ac185993584f2fa127553a43722f76db5bdda`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80a7f9806b34ab4aca7f2c918d805e0ef978c8cb5147a44aad086817dfd7315e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-228041d7888220583bc5ee106898729c662644ac8ba24ecd23382dbf545816cf |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4dec740c9ecfd577a90eb21b61779168d7c977478a8d85095e2ed015c0fcada0 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-756339e2a10b29a9d628ff8ee7123cc0126c3f49a694b158e407c73a366d3fd9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-304eb6d0 | sha256-6d62f8286a43d967346d8666cf138789bf91e82c6b6831346a0b504e42d596d4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-304eb6d0 | sha256-bf8415fcc675e77be6fa435e4cbc1acc4daca18b8c0161570ceceed9df4b0f6e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-304eb6d0 | sha256-6d62f8286a43d967346d8666cf138789bf91e82c6b6831346a0b504e42d596d4 |
