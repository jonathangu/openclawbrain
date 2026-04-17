# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2f2e67ba6e9f3ee34d9a729b960d4347b90c5776b36c8bb01215597777ac63b8`
- fixture hash: `sha256-31116913aa40fd67b6f1a05c1b62a0f72f8a386379a84cc5c256525c2b570370`
- score hash: `sha256-f581badac383e74cf937be4c7cfbbd3c934229e1e6d364fee265c5312c3616a7`
- bundle hash: `sha256-b4100cd5178c1b83a80c296b8447ed2f0cdc2139df071244272ae1d626d0889e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13aa42d069a6fbba4caba9f912ef9cadf19ea12093ab266f931b4282b9e22bf |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fe3f864f2a277b713435648f4a0531fbcf00380ef9d242a4dea00d1256c2e9ae |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bea0a5fbf7e4b73d890f66f3b37970dd5d6de96bc751189de6a11b5afc6140be |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-bf04a7d6cf5e8907b3da0084102b8af8b75b017be6fa9c563b7357e1013ddd0b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bd37169c | sha256-22f0d44d356eba134921c5858c319d7130ecb4fea6e4cf2c09ca06fb7c346eed |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bd37169c | sha256-8227c14f54ab29a1ae149d318013510ca09f88db970eb8f21aa0c9b9448e3a4b |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-80270f7f | sha256-abd2ec81eabb5feeeb855bc2cbf9f988433e7d154df40035148241530bbde836 |
