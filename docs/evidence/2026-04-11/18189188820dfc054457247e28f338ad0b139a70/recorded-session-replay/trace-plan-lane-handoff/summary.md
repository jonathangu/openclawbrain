# Recorded Session Replay Proof Bundle

- trace id: `trace-plan-lane-handoff`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fb2dfcc53391da243d2eb309363317b16592b228eb91cec9df07b31f719e0068`
- fixture hash: `sha256-f1ba04aa90e022e7a4283505f4deef91fe6f05bb8b561d462b99f13bd3455652`
- score hash: `sha256-64127b61e6d0f47a002f4191837b0dbcc480e1196a1c5a56c56883fe38b47465`
- bundle hash: `sha256-6916f60c8bb3a7d149847a95a232f33dfb04323f04be3c861365041327355ab1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 15/20
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/5 | 0 | 0 | 2 | 1 | 0 | sha256-91e1ef9d2b3d61940bcc1c02b134971c7b50bc3f88ccf5323436d9f8cd822fe0 |
| vector_only | 2 | 2 | 5/5 | 0 | 0 | 2 | 1 | 0 | sha256-c2811ca2daf5c04812fb0b29819068a0c4ed8090860c9e78d4eb9fda291a01fa |
| graph_prior_only | 2 | 2 | 5/5 | 0 | 0 | 2 | 1 | 0 | sha256-85fbf5d8cc467189432dbc13f246d83c2e6191bb5b7d372766996130a577d195 |
| learned_route | 2 | 2 | 5/5 | 1 | 1 | 2 | 1 | 0 | sha256-e5ddc8590eacca1aa9efd5ad8c76d5547834d640d13a7642e25bb48b0e3d9b1b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | lane-handoff-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | lane-handoff-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | lane-handoff-turn-1 | 100 | yes | 2/2 | no | no | pack-fede1527 | sha256-76321a811a7e1b986a573e41702bae4436c00af38a03be5ec858f83ec73ce2ff |
| vector_only | lane-handoff-turn-2 | 100 | yes | 3/3 | no | no | pack-fede1527 | sha256-76321a811a7e1b986a573e41702bae4436c00af38a03be5ec858f83ec73ce2ff |
| graph_prior_only | lane-handoff-turn-1 | 100 | yes | 2/2 | no | no | pack-fede1527 | sha256-76321a811a7e1b986a573e41702bae4436c00af38a03be5ec858f83ec73ce2ff |
| graph_prior_only | lane-handoff-turn-2 | 100 | yes | 3/3 | no | no | pack-fede1527 | sha256-76321a811a7e1b986a573e41702bae4436c00af38a03be5ec858f83ec73ce2ff |
| learned_route | lane-handoff-turn-1 | 100 | yes | 2/2 | no | yes | pack-fede1527 | sha256-76321a811a7e1b986a573e41702bae4436c00af38a03be5ec858f83ec73ce2ff |
| learned_route | lane-handoff-turn-2 | 100 | yes | 3/3 | yes | no | pack-5c1d2672 | sha256-518b97f98fcf2c403b0df24651b5653339fba3ae54d2e41afbaeca36606bfbc2 |
