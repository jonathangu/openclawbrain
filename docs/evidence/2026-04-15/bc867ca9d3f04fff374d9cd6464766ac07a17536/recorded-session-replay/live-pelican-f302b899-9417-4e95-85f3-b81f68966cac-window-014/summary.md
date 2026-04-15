# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a76e30e78a3c6628f4a84b691210152bcddf9b8fa0661b16388a6ca59daa23ac`
- fixture hash: `sha256-2f161d785d6fb80ca3ab0af035b3aa3abbed725f829a4bae1a60b67e83a88b19`
- score hash: `sha256-f67eb44c3cc8f7a8c85c63e36addc239ebb9ce52ba8e08f44c4138bef8268bb5`
- bundle hash: `sha256-207311cb4f4eb81e3c6f62801dd8eb70833c512419f95c96918d8affcb7012fd`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37705891572c574b5f2ed2ea56d6ec8c0372961de1b290750eeabdf9bb9948c3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-70264d26cd0152cd23eb7bdd138162f98789ba5febe5357124d959ff4c5b2054 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-78a43ba3a0327527514ae6f4459d5a1134f14f39d52e7a67a1ed95fe04c6b3a3 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e32f47ff5a3fc95fe7c6bd0cf68f58af11021df2970f0f294e7bf2e098273fab |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c9623020 | sha256-ae744b749f3ba55dc6a3f7d1001c75fa223e86acb767e47410151b216cbffa89 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c9623020 | sha256-4870c88dd07e6b4b447044a05688bdd67908c94ab47d78257652c78312256650 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c9623020 | sha256-ae744b749f3ba55dc6a3f7d1001c75fa223e86acb767e47410151b216cbffa89 |
