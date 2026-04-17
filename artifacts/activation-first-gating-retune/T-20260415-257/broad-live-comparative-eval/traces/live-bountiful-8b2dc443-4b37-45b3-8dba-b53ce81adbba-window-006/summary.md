# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a9f3db5c5e9f18aad5ca8aa8c8134dfac254479399202badc35306faa348393`
- fixture hash: `sha256-af03997f06ab50c99afcf76923b04c21e1338d145564c582674e59eb816853de`
- score hash: `sha256-3f3cdd121184b5e865b78a910871c95abccaad2b613e92efbb68afed3d374222`
- bundle hash: `sha256-61461b3cacca017ec08528b3a779d9adf792458e6e95192691cfa337a6188af7`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-048408a5bc1e1d56a6cc83e227b9a2958b83cb861b21925fd209ce4b8456f636 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-65445ce2ac2a3110532aa6426530dda90147148f525bef3000001bddd3f785a0 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-944112391fc417121d255e75b8c9e363a14b237517dc3e92fc42f8f804a3fcab |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-0c66af701ea0692e4a9a9f36773539ae3de8f26b0d7664c421461ffa086fa7c1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ed55706c | sha256-1cf5a1ef2a6433856881b2a2d752ce5b9a042b5d2fbaa028bf4412dbd426c31d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ed55706c | sha256-d6c4e96141cce498cdc6c248573f9b526626fef775ae3b6d9beee9250ac02af7 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-210cc381 | sha256-e2b024ba35f91bb4b1390bda7a993b063aef10acf0c1306fc7b7ae1696a0c937 |
