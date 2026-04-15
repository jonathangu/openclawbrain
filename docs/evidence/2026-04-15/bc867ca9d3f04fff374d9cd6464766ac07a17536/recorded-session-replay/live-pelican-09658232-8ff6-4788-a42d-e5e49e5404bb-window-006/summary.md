# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e1899273f160788957979a298d976827dfb7d2c8980b1c161a6e0c69b405f12f`
- fixture hash: `sha256-e3a4578dceff89673c40bbf12c9b294dd97be3ba2d82b9f266209970182a5648`
- score hash: `sha256-96cd35e9ca79ce91a26140768f69e70fe53d45e2e4db660763b40f4ba827a3b8`
- bundle hash: `sha256-e97e8f861e737586fc422fb8cb790e956dd4f3b7cc5ae0e635c168b5dc7dcb19`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ecf5a06a6508fbef20c40ee36944ffad441534c7ec83a389bf5c81a0f73bcb66 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1ca8a79f5f13fa11e06e69d5024e7eae69f71e545b8cacd473e1f4cf3fda03a4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c1efc93ecb60a1c97fcce0c35f500d3f625ff7c9e685dbec668083def9abb00e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-43f894006f60ea64b4805802a5709129200d3f8c106fc123a6ed136740cdda27 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-27ec3b56 | sha256-7aca20f387b4396131cf1dd6401df879b92c25b96a967ab43ee2836766b9c48d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-27ec3b56 | sha256-823fd1c17e6f508c2e4a7546a10e88f20b33494fd1397f8c7ee5719beb0db43c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-27ec3b56 | sha256-7aca20f387b4396131cf1dd6401df879b92c25b96a967ab43ee2836766b9c48d |
