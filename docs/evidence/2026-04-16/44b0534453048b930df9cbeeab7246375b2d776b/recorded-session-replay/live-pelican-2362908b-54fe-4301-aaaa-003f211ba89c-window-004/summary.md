# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae1abddec00632179423e5d665c773fa81ea75d92b306fc15251840d9f53ec48`
- fixture hash: `sha256-c2c90149661c99c58bd2b000a17d70b99f16ed3daba941c64a7e5c1b67ab99b9`
- score hash: `sha256-832f490fc03a9224022488708c802f882d75bbfcc45e61d78613e91c43375012`
- bundle hash: `sha256-249a72850cda1d70bf3bfee910bea20d85bd4ceb3e4c7821c668dd8953fc12c6`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-83c42e700005538dba5b3a6d69c6c5e443ab91af8b598837eb4ca6b5f8135237 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-11857a7a93cf6d5df1bb8224fa1587b789a7b386d48bb2880b9157fe9d794bd9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6ce1fb48d1e0d1cafb0e0bbf3e174bd023a14127bd20e7df47212c85c23d0f6b |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-8a19e85ad611715e8a3516d0c30bdf15c2054523a0d7e49394eb480c942a7aac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-beca5ce9 | sha256-eb3e9bdb84899018ce62b1480db49b66328fc14e4ff00e208133e00b3d2080df |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-beca5ce9 | sha256-7d7d519f1dd0569f9ea72e14b7d01e291d082d8768e880a8864966bd1a0385d7 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-1fdb0f5a | sha256-cc5208b32f5a5095b6a55ca98e2a58197aa93eb32bdb5de325da4e6f1643eb93 |
