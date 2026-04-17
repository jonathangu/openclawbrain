# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9b773fc52a6499e93a829bfadf5df1387acc7dbdc341067eb7448f1d1a77436`
- fixture hash: `sha256-7d2098e87edd7bd56201c0ed7a627280d7aeffa57ad2111b2d1c726f44c95465`
- score hash: `sha256-82a8ccc6d974c59b8acf7e7a507bd8a80f853da5f2d5f933dab8bd138a7aae55`
- bundle hash: `sha256-dd4fb68a0a061883be15a65bdbb44c4f68cb10c59cf4bb3b4a1f7347085058ff`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1fb773fe926bcbb4d8ac27947b3fb09ebad87653f896688c59fd65016d54df46 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-40796df71fa4660feff22e70b7b835ad5a4f34ab6aa448598f27c15cbdf3770a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e57c0623963f826b48b59841acffbe7ad03b2348aded595d66c85644b9d40878 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-49d7cf38 | sha256-d99aabd1b3c2826f6615144800a712a7890b8b1e2aec2cdbd15ce5009e1cb971 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-49d7cf38 | sha256-bf2a93b341592138e9398e346f192fc9ea09c2dcd1a3c20e54786f9d7638b9f5 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-33765b45 | sha256-469120077b3ccf40e6e484460b682ae18df416610ec25fb3fbccdbf3e97680df |
