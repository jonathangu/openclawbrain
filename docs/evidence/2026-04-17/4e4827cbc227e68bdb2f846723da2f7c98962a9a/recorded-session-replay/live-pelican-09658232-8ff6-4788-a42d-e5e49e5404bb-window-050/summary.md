# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050`
- winner mode: `graph_prior_only`
- trace hash: `sha256-edca99fed5bfa02f8196eb002c7ca5449e3ad77a55a7ede4db613e0216b9a288`
- fixture hash: `sha256-075e6ae40abca95623d0eaab9386a47facdde038c6af2c88f5255ae9a6184b2a`
- score hash: `sha256-9f344800fd22c1537d9e5ac4ec5cbf80c73072121873ec3f4dafa32d639a8014`
- bundle hash: `sha256-7f5fb203c2daa799ab969465138d866b96125fb25786bc919b29710ade9aa79e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6022525cea9be81df298416f8768122250fedda0d93e17bc4857c9bee2c2bbc7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b7f1274c3c5c639bcdca8926aaeefe4e89275cc1f44f7ba84f2eb0a9b36a88c3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c74f988f5ecb462f9a815c083749aae22ff7bd6e56fe52e2b4a53f2574b0c765 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f2a0adcf6bd7642be743d89aec05f0bfe45bc9d9d62f378dd4b69ffc551b4e2f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-1bab87d3 | sha256-8befeaf0acc2e6c52a4adfa6314a4bba0c5157c5a11df2cd7bbe1cc2bc418642 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-1bab87d3 | sha256-8befeaf0acc2e6c52a4adfa6314a4bba0c5157c5a11df2cd7bbe1cc2bc418642 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-43667cba | sha256-1a0f317d23ce73090c8d4de5c9e2ea85b132ec571794961c80f58ea37e2dbc7b |
