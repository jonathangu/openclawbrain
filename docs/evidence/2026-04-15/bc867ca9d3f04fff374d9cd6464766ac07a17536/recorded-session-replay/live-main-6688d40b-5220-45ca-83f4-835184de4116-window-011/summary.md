# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-850bdaf9d44fc5882480e1fdabd688dfe420007059248f68b1bfcdb177c8d991`
- fixture hash: `sha256-01700f6ae7fa9661baee2d1698232fbeb6cde54e151f8324fe1800456806d50b`
- score hash: `sha256-1b37bfbcc0f0d5ce1c70de7ed4ce24b50b882df46a405a8c8c8c583603fff2eb`
- bundle hash: `sha256-f9081fe7256716d308d4debafe7ec31b918456fbea414253d814ad52b3f32396`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-db5ed18ce329dbd2d8fbf4381eae760a575d1622b5dffa25a4d7dabdc4b4d367 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a51f8cfe00160d8b595cc8530614f6f15898d9a61fccd9c3f6fc26cd51dabfbe |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-03d2b6b6a0e221f9affe49f944c0be6c6633df42ea6f5578c599a8d4a078027f |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-afc5a0a834b8fd394ddde363b4271993905ff0a374d0adedf5cff1ed830ce75c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2eb16041 | sha256-09586b74e57cb69cfc5aec404f2f9a22280474ebc4cc452640d025183db6762f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2eb16041 | sha256-3426368a2677b2b5a69ad6548af76fdfa87cb3055e3a1fa16e415d97cff5c709 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-2eb16041 | sha256-09586b74e57cb69cfc5aec404f2f9a22280474ebc4cc452640d025183db6762f |
