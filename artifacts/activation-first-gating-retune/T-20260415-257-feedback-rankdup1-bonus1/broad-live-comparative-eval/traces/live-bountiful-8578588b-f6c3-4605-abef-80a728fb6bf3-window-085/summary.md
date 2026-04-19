# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6777f43938f352c98ea100c93814728b861aa4c79088c50cf000c1d5e433fa17`
- fixture hash: `sha256-2377709ab100f5b5757cf6a5efd0292e1e4a82211883d9948620159f3fcf1f8c`
- score hash: `sha256-57fd2fd5b7a73c7fea4bb4e9fa27430190448aee44e73c09bb573fa68e2a5202`
- bundle hash: `sha256-f4b834adf337ecbe7039b8e2b61e4a083f3e9a9e48b70f886cb58c31c3e55289`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d40f6b5cbfc3f93a14eec67dd0e0db8991d9a04dd6b798d88e515d5466bb7a07 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-5984a7500d7c21cf30b2f8daaaf1fd25993e31d5146a3aca8cf36c6aa2041c21 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-7c039a8800e47e523e87a32b44af3a115691fae3dde2c142fc70f2989e59541c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2d357aceedc52a4809a875be69879fff761dbf2dd1a097f7eaae94d89a6bab8d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ed3ef03a | sha256-c336b2bc944b9b9b73be2c2e0dc331c8a9bf0bf3ac84e91b1c1b3c3c8a118fd8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ed3ef03a | sha256-821d1567c3c774f6cf03c04ee06ed4aeeff963b2f4822a31f39db7e55eb2dfb3 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ed3ef03a | sha256-c336b2bc944b9b9b73be2c2e0dc331c8a9bf0bf3ac84e91b1c1b3c3c8a118fd8 |
