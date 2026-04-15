# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8b83288abc1a5c66a218574e9a089abcfea75ee1de4f5813fd07c339a4e34fa2`
- fixture hash: `sha256-d84bdb541f6a2d5c8236abca3a843aa21a0e1c20f003d0fc5eb1d79b307b698e`
- score hash: `sha256-706bdba7e24aeb0d62d04f88634438a3d1652e3d0f083a2ae8b95de8765a3d34`
- bundle hash: `sha256-cb7d23fd2ff6cd7f062493dd353a1f41797aace966c9dc623d222c25a6b38420`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7ad0dcf523c4d76bf7e5aa9a9c949e660e04aa89d0cc57603f9d8d3b2165caa4 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-729a1c00f462602522917aebc56fadfb1dd638a7c73fa2d66d1ea99be40ed04d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-60a91a446c4f405856892078110d39024023ff949aa1b4048fc0efba0a29bb60 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-add23cb0044117f01be4dfb2b1ab232907a85460c4cb8b4ddfde57e9c84dbf5d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-096bc85a | sha256-d8e1f0f4e3df3ae3e1b4c27f9e13e05800a919bdad3ca1e54ee20a8fb80a5943 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-096bc85a | sha256-9249385208fdc948752c5889615fc08602e03e73a69576352cc49269a736d813 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-096bc85a | sha256-d8e1f0f4e3df3ae3e1b4c27f9e13e05800a919bdad3ca1e54ee20a8fb80a5943 |
