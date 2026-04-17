# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1e91f891be11ad983e343a9bbb8eb7e094a3203fdeb0cba32d80844dcceadc5b`
- fixture hash: `sha256-c962d7bf59f91132e81f529b35b43a46128d3cc144f19a803783e383eb2588e0`
- score hash: `sha256-f6bef4bd2f535484bf8f65ff83c2a42240912eaa1cdb1b63658f72ef7aedc232`
- bundle hash: `sha256-61f64cd38ba3486959826a3d68e43a50a14cb06e8ba1606ba9d4fe83b57c8573`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5abc89ba1c4aafac24d8b492241ea58c50f7925494e6166e3016c9a753e61584 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-14899399347a1517b8e93cd6960f7e8b6262349eb74ab1d8d3b5c3dc988bbdc5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca3ecbdb0a56d1cced8647723ed755ad8d432831eafcf92c45d3e136a005cebb |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9330ce144463fdb7f5443dbf6749af1c8e779624b6b2ba29a3482f43bff8bf30 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3bf1082c | sha256-72c1d3541161fb78b8e73684fd75df66c582d2864c007d8026c85e41b3a8d787 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3bf1082c | sha256-f66a847991fd5e441ed0098fb670c601bba92c7af69f962bc5a58d4d698f883d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-476937c1 | sha256-b8ae6faae0b47ab1b50eac438e57f2da424a74d4eaf1fa632ea23784e76e706d |
