# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-850bdaf9d44fc5882480e1fdabd688dfe420007059248f68b1bfcdb177c8d991`
- fixture hash: `sha256-01700f6ae7fa9661baee2d1698232fbeb6cde54e151f8324fe1800456806d50b`
- score hash: `sha256-1373918e46274bfbb084a4fd57b87d82db781143459db5c1156b7f80a556fa05`
- bundle hash: `sha256-045b83862684fe57eb3771f3692e92467dcae58a48a6d1a103e7ac705a4ffe00`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-db5ed18ce329dbd2d8fbf4381eae760a575d1622b5dffa25a4d7dabdc4b4d367 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-97cfaf492f486f109af8167e6465643fb553b344debeec8b30d06c69dc058781 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-320e70192e61e9c546477cdb2a517a9dcee02ab7375423cbb6e03c862210548f |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-671f4c74b8fc27658c33b27c8f1638df104b0efd95dcdf9e339b9ba6658a0adf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d46e1d3c | sha256-99dae607f9a4f918d95dd99c2ddf08c6c5f5723a1e67a2fbe273636a6cb79589 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d46e1d3c | sha256-bf7db87849a43386829b0b23af5dd64464078efdd06266b188010769998c0128 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-6869bc57 | sha256-6be5a37749e61f055130ced57f2bc88bb48d4dae7d7bd0a32804fc98ddd1a771 |
