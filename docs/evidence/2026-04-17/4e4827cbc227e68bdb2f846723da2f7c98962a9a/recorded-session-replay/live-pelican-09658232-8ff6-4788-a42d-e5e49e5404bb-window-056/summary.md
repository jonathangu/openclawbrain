# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b194659082568a82c511a7152ca31a2b1b95a8940775e0d8501ad2641699262`
- fixture hash: `sha256-818172b532c3157150cdaf4f843fa921402c9f435a9b49f1a0bba05b616c0656`
- score hash: `sha256-fd513090afb294e988c697b8ff2f03acf7e970f35694e2bf174c3cf0c86f4568`
- bundle hash: `sha256-612a59f0d37a4253ad440faee93de025cd323f3a439f90bf867646e509b1d846`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3c006aa8496cda3a74dc0aceaf43d36eb374dd8330caeef238bfc730df80da87 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2f0481663372e2867be2d328453c56e7b9f947ad86ca2dd613ff546d7967b9b8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-39fc3e5ab8140ae8004530f39a0f56f4ad41f902b69933e1710e0ed28e767af0 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-0afc77bd2261b482c2d2e261253cc1d84cc15aa02d14831531e88e993415f004 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7ea80984 | sha256-70906656408fb1709dafd7ad000c38d938199b7d6c3adcd9885d20f1452178e0 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7ea80984 | sha256-100bd581f57105e1a6a00b6ba9722d1410be4df99de547701030fe58fc48dc16 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-c9d8680d | sha256-66dfceaf0416f0af0e3a88960cd12959dcf8f7983dc9b1502c30114a06239412 |
