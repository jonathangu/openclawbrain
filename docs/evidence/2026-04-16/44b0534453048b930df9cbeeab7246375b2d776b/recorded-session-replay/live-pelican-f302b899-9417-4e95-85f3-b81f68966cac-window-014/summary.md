# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a76e30e78a3c6628f4a84b691210152bcddf9b8fa0661b16388a6ca59daa23ac`
- fixture hash: `sha256-2f161d785d6fb80ca3ab0af035b3aa3abbed725f829a4bae1a60b67e83a88b19`
- score hash: `sha256-8b6f1a3f973ed90c443de1aa87e13656eb335f24f1f99523324582d3ee80e43b`
- bundle hash: `sha256-56c3c6f2e23956cb64aa1de94beaa7f9164098ca447ed7fcddd5269ad333999a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37705891572c574b5f2ed2ea56d6ec8c0372961de1b290750eeabdf9bb9948c3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-df802915eb6d02374245ada060ff519351c179a605d02d3cd60b3ca40d3a9c75 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-83aa4956abeb6f8581e39c282b80c1231ebd63474900d4c41f33c90e5ae12e47 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-3e41daa0b88bf17bda46836aab7e4e9b63255c254d8c61c3cd9ca963d434acd1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e68508e8 | sha256-71aa73ba495bde1965627fedb3a75c51635b3810f6558d1607b1fd28fa03f8cd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e68508e8 | sha256-8efcfb15ddc7fc29cf96dbe2fe39fb0b2601003caa61bab353c1bcccb5701db4 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-303b81f7 | sha256-6e6595b519544200fb059a5af58a188929a130808b277ba1432683a3f3a7be7a |
