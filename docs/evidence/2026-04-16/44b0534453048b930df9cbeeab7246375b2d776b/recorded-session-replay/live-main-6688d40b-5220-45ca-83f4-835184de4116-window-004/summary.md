# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2210bd8aa54ff55f81e90c13af23591578b0c820206054d3d91e01211b88bae7`
- fixture hash: `sha256-a562aa7a1ac863aa823f236bdbc816afd7b8d62760a47e5474f699f78bdac5e9`
- score hash: `sha256-90eb74fd151d818f9068bbaeff8f49feed2e3a6f3985b71f17199c57a0882735`
- bundle hash: `sha256-847e0a4538bdf42e9bcbf9532edd39e79eb463e3f56e6cd8ae4b3e7381401b7a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cbbf8eff09f23d982b9af94fdc9d383c8e6e748daa65afe086a31e073a634311 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8baa74a2eed2eb61a5cbd0bb25f01eb58862762e9096d64b1f7ff773129d07d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a1f3aaf8414c17744a147ee447254320b46daaa9b05e63ff89f13e0b83beca3d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c12eef8ea3eb2b93f68b5f45eefcc89d333cce9a67cf32952b84fe500fd0e53c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-01b9191c | sha256-f72c0bd3c4d100f08327e26bdc99179a2e0e316fd08b6fcfdbf3556ccf693414 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-01b9191c | sha256-5c359834e4094307c10ff1f5aca4cdf4d3ae76c378c28210e5de93836b038db8 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-febf4f51 | sha256-ba415840caac5b277d75f928414d0448dc56c45b5128cfcbff99c26ce2277212 |
