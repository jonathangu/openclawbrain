# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c817aa28a6ea88ab750b90d075966003c4144ca68cee4de31510afc8940af725`
- fixture hash: `sha256-12c8924300be23df2d629cf06b8bf4e9466d47a9b90ef4b0770c780fb827282c`
- score hash: `sha256-e86b520fd57bc13e591640a58b399f8663cf0257e66ba7c9a937fca68280591b`
- bundle hash: `sha256-2477d7924f97969781422a49d8831a1dbff8a5ddef233a18dc34e27c74ea3471`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a755e36be001d38e08b764d65e8f6dd1b01494428975ffb22d7f3f721a73e79b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ac553518ce4c74bae209d0ea14de80d5a37ec510132a29674652aa612ebce2e8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5a9017446f48275829792f92867fbac1f8784ba63643edd7ef624a1f50f41a50 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7a253036cd4d79479db3c74fafb4bea7afbda454871b92129fc8b5e82ab769ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5897f2f2 | sha256-4f3a0e33e3099af6962c98b451f156d6b342130cb872cf190267be9ea7e06bcb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5897f2f2 | sha256-4018a22a1977762f5c530f8f8ffaa6e8a93c2793557e2799238f9d94252c504c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-be1c95e9 | sha256-4288af0b56ff1df6689bff2168825d06f5431443e962fa9ba187eee195230980 |
