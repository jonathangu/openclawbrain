# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bc492042fe348d82c21faa673d938d0577346ee06614b38f34d614d883fe125`
- fixture hash: `sha256-166379a9c9e98e60de3e148d45fed20846d7dac8b779bfce9e0299ba405d4f98`
- score hash: `sha256-08615c252807e63fd4d90219f33d38a4e0cbf297a6231573a8f1e258af305837`
- bundle hash: `sha256-3fb9532757fce3e1bf516a5302cbcd2b831a6e464020a84019e5e503cc57001d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df7544610c7c12f9cdd0d8aad84f983991755b60031939cec1112c0295581782 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3166796cc2485e72317ac0c8f7459a6abc9c8c9c0c22c4b9580be6fa39d798e3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-29f7f940a3d50a46682657ff9ea2bc074e6bf1ed7320d8407e7764c68cd3035b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-446e8cdcb62d2dc0fdde34551a15b0e5d506bdaa3035beb84630362e55b1d642 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-21533fea | sha256-bf60d86c7a6500772a10b789b909926e44c76aaa2d5add626d03f160d760b3e6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-21533fea | sha256-d5fa2bd4eaf426bc023d1fca9b0199401b194c5da9fd4a49d08b007f1d9d2a5e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-21533fea | sha256-bf60d86c7a6500772a10b789b909926e44c76aaa2d5add626d03f160d760b3e6 |
