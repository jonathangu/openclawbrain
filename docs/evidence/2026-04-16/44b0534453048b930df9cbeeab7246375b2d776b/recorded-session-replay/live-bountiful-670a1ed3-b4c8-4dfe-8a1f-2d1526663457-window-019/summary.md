# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3d68f21a3db07e083abd55cb8f30309dffa35aea63874e95510f19d0d69cb1ce`
- fixture hash: `sha256-370af296b8752ce6655fe59921b05e957209333f8adae37b056699cf10a9af35`
- score hash: `sha256-4d4074b8588c74eac659a81b7573f144d19039415b156de1512299340cdaf488`
- bundle hash: `sha256-74a0f5ea786dd7ef9ebcaf6afe52566aecce02f84e07aa8d0059b451b648255e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13ac363ce7285d3640914d39071894fd6c80687f14f6807f8531ccb47249088 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ac201052981265a27cd8936d5f5fb21c82d4bf2c94bde13d3000fbb75d165db7 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-39121ec85a61f812c77821e01d69d0e320ec13b5362a80ff6449c44a8c31995e |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-216ae545fbb37a61e06295fa234718e64fee4945935417045a1fc4841e44f020 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7a0a5f8a | sha256-4d27867163e49d970bdc18ce9005c9713d8587c1982c14743e79096d7a1ac4fe |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7a0a5f8a | sha256-beb4acdfdc8bf253fbc7db6848d913523df49e0315baf189455f03200d7e4341 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-88282715 | sha256-8e834d068435aba3fa8c875181534f70aa5c2f25dd870402b9608f2bc88aeb0c |
