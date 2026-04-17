# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60561f5cd4b9679d1d07ec70fb93c8ce09ef36cd5a40b0352b67931141e9e246`
- fixture hash: `sha256-d68e77ff5a53346b0fae859928eb6131851ab9f7d88f52a94509c0f85b109391`
- score hash: `sha256-b9dbd9eded959b0374988277e6a4e866aa4c9064d9dc4241a225beecb0a82b50`
- bundle hash: `sha256-be6cf4c5414d45bc97feb1fc776d9ed77878035088d35b7b1acb15eb74725d6d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-350a961987884e875cb36c29ae1cb810ef961abe38158c92bab3e2c95369cbcc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-87be92519c33f11074de4bdc5c6d18ddc729dec2b2dd55d9fefd7e95e86c3383 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-639f1a8d61618f862fe7de3d322eed9d9088ed076addd8f59eca39debda81bfe |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2d5d95392f19609b069d7f9bd3c5dc5c26387d129bf4f02b16d12edd3aadd2b4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e5ac2147 | sha256-ed58c57fe6c1d26a45990f22039c18f0f456d7a8945dcd64318c811f58756146 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e5ac2147 | sha256-e34da27fd2c96c5f14ac94c5dc14916572b0b252bca3244b68695ede62e2b92d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3da46294 | sha256-1d67aa0681c4c99b072ab7666a2b29f8c3def9830c41c9a2437cc82750a24864 |
