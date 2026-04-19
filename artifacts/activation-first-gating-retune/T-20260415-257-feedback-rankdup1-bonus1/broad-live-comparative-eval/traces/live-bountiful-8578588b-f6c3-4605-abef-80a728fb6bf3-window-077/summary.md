# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9b773fc52a6499e93a829bfadf5df1387acc7dbdc341067eb7448f1d1a77436`
- fixture hash: `sha256-7d2098e87edd7bd56201c0ed7a627280d7aeffa57ad2111b2d1c726f44c95465`
- score hash: `sha256-b3e4c92b6f06b97d045e999dc1bc20539ed12231c6502c63719fae470907fb79`
- bundle hash: `sha256-57f339792f7e2a71646683954997c76da03dcd3a25ace143f53a0546577b98e6`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fa853751dfe24e892b2663eafad4951f4882c30930db970a96779bceb582c370 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-9cf13070fc9f28b3a4cf5d8e23e9c7ff07fa15f345139c93b20a8b6f8f9c0e70 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-cf305e211082c96c4fe7c2dc9d8935b088ff68b077883df94c61510a286f07a8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f37da4ab3ed06fd805575bdc2094311b54f3863e9503b5caa1b2cdc74da7d7dc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-33765b45 | sha256-469120077b3ccf40e6e484460b682ae18df416610ec25fb3fbccdbf3e97680df |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-33765b45 | sha256-9d6e828e9fa74524fa6c3da7b01e5b45c46a946e1c6a318a4373abb4583a1f73 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-33765b45 | sha256-469120077b3ccf40e6e484460b682ae18df416610ec25fb3fbccdbf3e97680df |
