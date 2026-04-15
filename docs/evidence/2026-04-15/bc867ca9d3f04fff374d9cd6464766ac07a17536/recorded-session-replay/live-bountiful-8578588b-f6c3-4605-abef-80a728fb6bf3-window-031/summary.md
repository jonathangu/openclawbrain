# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b95b3f2d8b9d73f4ca18258a6e6b859d4663ae6e1dcb8adbc358ccddee06f53`
- fixture hash: `sha256-e947103eb16b8507ee81b448d2eaccccd37ec8449f02f552042f6857d04bf6f2`
- score hash: `sha256-a994a11bfb0ea504c3fc1c0f08ffab30f778749813d6a691d2c5193d2b7da5bb`
- bundle hash: `sha256-0b1dbca9fd3b171f321bde8d5c550adf2d1292d20796b9a91f9ab2376ba65eea`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-45ceb0258ee7c7f1a993bb3ba076165647abb9bb280ad94965584a6223415962 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-52df712fa053570f5c54674f47743e973c1631240b43aee9588e5ae366eda09f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d0cd0b3c7e6080769bfaeaff615dadc6da437c55ad3352e7c6f9eb9a08d75e99 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-2626234bb86ab2ffc77f016f384fa34748de5ab5a53cd22864d0cc295d275ddd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4de3c73a | sha256-7a5aadc11f78f81ecd19f44cae74d37702cf25ab6483a113e820983828e67abb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4de3c73a | sha256-40c0ac00840382aedcb9ffc6e03e7e12bbae7c1352009dfb8031c75a570ac36e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-4de3c73a | sha256-7a5aadc11f78f81ecd19f44cae74d37702cf25ab6483a113e820983828e67abb |
