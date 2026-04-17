# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-850bdaf9d44fc5882480e1fdabd688dfe420007059248f68b1bfcdb177c8d991`
- fixture hash: `sha256-01700f6ae7fa9661baee2d1698232fbeb6cde54e151f8324fe1800456806d50b`
- score hash: `sha256-66b90acc9da9dfcca93110269eb235d60f51ac01b2304f72956e769596e7454c`
- bundle hash: `sha256-1761ea5356bbb2c9d8f4376f60559561e994ed13df5457d61191d61c71e03de7`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-91e661a6fdb2f2884731eeaa3282aed2e6db57a35075affb28174de669053fba |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b065355c7260749d53a3fd81db7f66fb2cd301e2d40fdbc6c1674d84c39149d4 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-e9c77dc94216c455b2d0fcfbfc204d677bec757ad82c9f02004ece567b91c9a0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-22788fa0 | sha256-1ce8654a764e0a5fd2e40378f16dcf20528785966baf88ff3e79de23972b011e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-22788fa0 | sha256-57339b3f7a464cd384c56e6f562fefb92f354c599e121d9f68e406f0b4daf888 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-b6742ebb | sha256-3f2cc8ee06ad6d48708996a2bdd18e4300e8fab263aa31638d65d4b5e2b6eda4 |
