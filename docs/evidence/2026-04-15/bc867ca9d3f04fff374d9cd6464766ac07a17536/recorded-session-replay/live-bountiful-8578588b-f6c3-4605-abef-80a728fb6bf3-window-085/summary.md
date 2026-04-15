# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6777f43938f352c98ea100c93814728b861aa4c79088c50cf000c1d5e433fa17`
- fixture hash: `sha256-2377709ab100f5b5757cf6a5efd0292e1e4a82211883d9948620159f3fcf1f8c`
- score hash: `sha256-4adc44f4c8bb4748b508ea550e4cab1d5089b4fc48b1cbd89200b09cc784d99f`
- bundle hash: `sha256-42be2d86793071924dafe6e6c8dcba33d5d16810047489c1af3ec88957d67dd5`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d40f6b5cbfc3f93a14eec67dd0e0db8991d9a04dd6b798d88e515d5466bb7a07 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2ac6bbb7b9556e1cf558353eb16085cf74764637750fa187f3133d740924ab0c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-25e80adf026796e6bd1944d063ed76b1db0bfb3801d9b5737fcc4f6811c6a828 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-11f381c7b5bf7acb3dff233954e058aae7b78cd77ce7f66d8a2c71d1e1b103eb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-36c16977 | sha256-82bc70e853efe123ba2a2e88c44388be4ce5774e9fe53a710c4e87668cf3957b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-36c16977 | sha256-a7740133ed8f40b3f5fa67e53f8d917910d9dfeae57a68fed4ab1524fc90a257 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-36c16977 | sha256-82bc70e853efe123ba2a2e88c44388be4ce5774e9fe53a710c4e87668cf3957b |
