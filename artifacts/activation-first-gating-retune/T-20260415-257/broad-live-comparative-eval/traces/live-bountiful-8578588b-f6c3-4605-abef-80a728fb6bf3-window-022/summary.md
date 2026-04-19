# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e9723616edb2ad54551f6589a0d45a8a37518248db0b8e108b75c34c56efda98`
- fixture hash: `sha256-05b8c1caa5037185047ead07b4f318668a0cb8dc8aebbf981972a18dc900efd9`
- score hash: `sha256-3dc4ed6031bda0d93eac8be6ce5c433663bb2328c4f29e53408efe89b78c8204`
- bundle hash: `sha256-318d53c011a4bbadb3723cc2af3b1a33eb4ae7137339a50990f6a31ff4218d15`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c5680042ae3bf747adf63e364ca5bf29ca561c697a87cac9ec59524d5a5a73c2 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-2d95a2b603621d3c1db1ae683a43e5022db0777ac879e95d759430bc5c332975 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-8d74fc720591c072049fdf143a3b1e02aaa8f64e95d12d533973d352b6d653c2 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-a56d53bf95383908747bf54c4b0aaac114519939bf83040ee379725432a8d0e7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-b27c494b | sha256-3a582ef0b986a0d26ec322a25b517c0b0168ce4c692b525376bfd087bd78823b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-b27c494b | sha256-8c64c5d9d84c3334e4079842eaae350f978e9fadf0829d0918e62e97dcc9a811 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b27c494b | sha256-3a582ef0b986a0d26ec322a25b517c0b0168ce4c692b525376bfd087bd78823b |
