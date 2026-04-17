# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a9f3db5c5e9f18aad5ca8aa8c8134dfac254479399202badc35306faa348393`
- fixture hash: `sha256-af03997f06ab50c99afcf76923b04c21e1338d145564c582674e59eb816853de`
- score hash: `sha256-476ef78cc081fd39b903f99a8b297c9532c1ebf25d34e98c8626f52cddf6f4d4`
- bundle hash: `sha256-69dfc991268c3188ca8726baa678e158807152c1db8f7b40637676f224f7c88b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-048408a5bc1e1d56a6cc83e227b9a2958b83cb861b21925fd209ce4b8456f636 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b3e4607c09c0d214a1e4511ec7fe4dc474b081c6e1016ee4bb98401e4d17dab3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4e9a9f7c815c4ac1acd36b77f1a6677817c6325e3e57241889d356c61b3f984b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c3aeed57cd4d521ce19c13156840dd6aa761fe736ba70d34a78c2d88c3bd0466 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-98a61f55 | sha256-9890ca23d4d3289f9e4f4b1b0c8c974fe34259ae0bf456f3ccfb80cd0464951a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-98a61f55 | sha256-9a78dbb224f83301f626907028374119027ceb5161cd832150f45423869735ae |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-cc5d726a | sha256-0cccb3fff7a94c0df66377b796be7acdf4952a0a5b12a99add14f82599db7a7b |
