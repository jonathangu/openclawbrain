# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e5cecfcda3863d55354a9b67074a3c6ce69c277ae2b3137a3f72bd0dc80700f`
- fixture hash: `sha256-6958fe867e36da1beab1df863be77bc3ca8278fa4e3d5aeb7c88307e08cb7f39`
- score hash: `sha256-2235d554c97d955de935d1964b8eb80086908983aebcdc3b01c85d90af3fb5e0`
- bundle hash: `sha256-a08d13a7629c253887012b05f908bd14579bd57835d62000e0f584ed9cb7a049`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7381932aea4d1bd30c10ae36d19326006a8cb4cb3b6e5b2b2ae6dadf03b6d135 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d70fbb25cc3bd7a141d3457da6d5dd4bf45a1dab5b646224a2ed6b9d80798c7f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c72511ce72ccf5be586b65a89e0b080bc4f538fdf962defacc643eec26594985 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-2b511f9f6860b0fad1216eaf6f0e2f6032913cb99ca731bc42f82303e18abc10 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3846d18e | sha256-c0df985c3b862980c6ac4bbd708a38e74f074e9e84329fb1f9bbcbb1796b7a4e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3846d18e | sha256-b04c1e60ba1f36f064e808e7a7ccc97d8a8d71e0e4b809df026074253f20dc53 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3846d18e | sha256-c0df985c3b862980c6ac4bbd708a38e74f074e9e84329fb1f9bbcbb1796b7a4e |
