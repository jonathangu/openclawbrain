# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-766d9b6ce430d9d07fe2ff3297e9849fe05332d7539d3d62db1cee2a9f89081d`
- fixture hash: `sha256-21e8a90c2dad8ab78ca636bf0f382e5b550e2af76a7681917f1773769c731648`
- score hash: `sha256-81b0529fc6c4b2571c7d377402d51873de1e4c8b9ff62c810391a2ae9feb6d4b`
- bundle hash: `sha256-039d223d188f6bf02c6b0c3aead7196566cabc91f375e8564daf971b9e17517c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d8021a8424a98c9c0ae913d23bd911fe66b4179fa226e5ae4873cee34e53cd89 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d086a24180d6b5d51e0854afd5a428d935e811df0b5d965c2c9249b8dc85d20a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7a63dbb0c8e33bf28004933c66b3eecc499b501206cf1fb5bb7e4acde28777b8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-baead9c0a335e0f0d2210555b964512e54cde145288b5bce3abad2d463147500 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-97c5a474 | sha256-e180e012c2201b2cd6dd1441724b2bb88470d65d15efd151f1f2055001c876f7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-97c5a474 | sha256-6d3231f783afed8be99e14e6ccbd80f934a0fdc294e053e1dd5c3106bd411801 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-edfdf8e7 | sha256-169bcedc3bcca6e99a1f20b3377722f3b618a0f70e7bbbf46cddbb6d748a2056 |
