# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97d7d39c8ed80340fd41820d6d636bdacbec2fc0c19c6596d376217775b20481`
- fixture hash: `sha256-cee22d0c8692c9c54ea684f49e1d3ac5076518c4157aff7a2d52bb3e3278c63c`
- score hash: `sha256-957beca3c5abc4bd86e226a000bd465a90b5dc7c1601ce239e3f323be8adda90`
- bundle hash: `sha256-ea870e72cf32b6216ecdb0bbb693e32be1f013b2f118356016fe4ba698848a9d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37de16724e3f909b52770a9de834272378dcc6d8dc93db3d2e32057318f060c6 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fc33a91c9e5cf1a6d436a2e7bf99bc314f1a028bf3aac29a17bc96436e10255 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-de42d1fada6338b0ade348bc72249730eeeff3e61db987b894203410ca49decc |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ac207f627df64d55f7921ef8ef893fa711ccf8e0e18b78dbd7758b6e3c788a86 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0bc9dab2 | sha256-a06d0f00ca8c93646d1e6f813751f80a9e59dad69936db7db5b4a8d283541e37 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0bc9dab2 | sha256-b2bcd00de56c8f681dd515831a54c16bc6e6cab2d799655a1dbd7d00e1e8dc53 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0bc9dab2 | sha256-a06d0f00ca8c93646d1e6f813751f80a9e59dad69936db7db5b4a8d283541e37 |
