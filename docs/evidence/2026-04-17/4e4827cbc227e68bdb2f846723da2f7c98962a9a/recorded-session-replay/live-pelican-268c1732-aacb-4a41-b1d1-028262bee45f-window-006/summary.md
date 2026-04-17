# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fc1409b104d617856751474f01593056b66d1b2ca492e8f5dd879839efd10f66`
- fixture hash: `sha256-8310747322d42de0fb2d06597a429aa5eb75a2026f88cf3e458dadef80911084`
- score hash: `sha256-33498031bc925c0147221053c5182a85dcceff2a5ce2559a35d6874afc9af1d1`
- bundle hash: `sha256-08d9f8dc2481c284f329a22380f3d2e2206b309817a1bbdc5b6f9ff5244cbc9d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0fcb97b42b2b441ec8190e1bb06fb82b8bdd1457d8fd6d8d105b2684066c5870 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-31404a7182e42a4b312d8a49747efa2677552574a765dd5ce21a3c57e83113a8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2c63e8be8f3f8613969fdf2eb7a06747a6f620294b94e34439247dc057365859 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-91591952028d747aa9250b1015883c9c583a6d0b8c756c0e56f8223f7a93e57b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-233cce7d | sha256-472fea9a40b16b1ca7923c7600dfc38e1750de044e5885fed7d24d0de0c9ce15 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-233cce7d | sha256-5a47985dad87bf254c7b7f3c652315d306cedc0e63b32f878d75650e5188e43b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c63dae8e | sha256-1fec413e7e8c47953c4f7bce904f75a97915c2d1c09a6e8ece94b0f6bb55b90e |
