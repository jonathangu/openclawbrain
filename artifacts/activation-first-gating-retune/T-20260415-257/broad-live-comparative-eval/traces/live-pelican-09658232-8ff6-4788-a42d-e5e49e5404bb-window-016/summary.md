# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0e0ac143317067e59f64740cdb9f819c48d2981153767f573c0e73b22b2b7c81`
- fixture hash: `sha256-dbbac8f5cf8c52842e2689d4f90634fa33bc0bae1bc0d3bfd9ad2ad85d720253`
- score hash: `sha256-db972f17752f455de48a64ab54a2626445fa153c5601da5648336694e60ad840`
- bundle hash: `sha256-4869acd71550bf0b4fcb1ff2b3aa9325838ef9c1f6611230bd52ca2c403676ff`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-745541acfd3bce8c03c831feeecff054c455963b939319f1092513f43c7bfc25 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-d85bbf18206580d38602b33a20e6aefb1543fdaf79a1c12448346d9e90edf5cc |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-6f738a9db8f4f46c378a2fe5dfd490ee82eb70a4196496bf6b23ce23796152a2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-317f30becb56b7cb7bcf22b053158b28719e297749fb4dbe1155da1d5ad6db26 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-c0bbe941 | sha256-98d038e18058eddcaf52e3abb2c2a6d89f5cc8c6ebaa0ebe1e096bebbc5adf9d |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-c0bbe941 | sha256-d7ec7b55ee0b249a5c653ac322c0b0cbde3322144bf0cf6efcc3b0acaf18f4e4 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e3732764 | sha256-b2dcadc8856d69cabb803b8f2c488625887e1c8657731c1d5d3657c7ef3e67cf |
