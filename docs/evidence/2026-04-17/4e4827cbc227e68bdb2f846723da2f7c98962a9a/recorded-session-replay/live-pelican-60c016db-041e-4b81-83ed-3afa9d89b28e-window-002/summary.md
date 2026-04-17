# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cfee0e7c7d7a332b7f5a4f24cbf19b04a7b31698d12d254a92d62753684d371`
- fixture hash: `sha256-539cc58588045f4d44638a17795295875c8ed45ffa9d4d266b2c19df9a95dd7f`
- score hash: `sha256-dde8e9da0e72030779f7dc189a53f145bc9a441cc3a59084a0eb57a0c6acc827`
- bundle hash: `sha256-8ce3c707df67bf8e3b6b984df9fe2e121994ae6b00539506f74cdc555abc80cb`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c32ce668e6813134ccd828363a96ac1b89f56519737480b00962b4a14175506 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0b7583cbc5c3bd01765295068e417595ac88a7da89f852b31a6d95af4dded6e6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-260a5108d6246532d6ad74b6f1558627529166fdaaa2545956a62d3bf1c2f2f1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-9f6b332280090d3ebb9200922aee5370d9c415749b158f2b0762524b83c3a9e6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-338e30d1 | sha256-13ec65b1638f23f081380722bee9d1848cc2145cb636f14af6610d9041a6b6e1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-338e30d1 | sha256-6a75e5dabba9d6f20c70e3a2824241f8b582ca3a266ef180580cf2470d472ab3 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-dcd8dcac | sha256-d8ebe13dffba3e99fe8adc35993f6c464960bbf620b834ac2089b34386264daa |
