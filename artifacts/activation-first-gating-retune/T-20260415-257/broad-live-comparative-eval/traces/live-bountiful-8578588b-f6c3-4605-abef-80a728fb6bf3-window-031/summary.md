# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b95b3f2d8b9d73f4ca18258a6e6b859d4663ae6e1dcb8adbc358ccddee06f53`
- fixture hash: `sha256-e947103eb16b8507ee81b448d2eaccccd37ec8449f02f552042f6857d04bf6f2`
- score hash: `sha256-88a61f07c248675c9c003f647a5f2ef476b641622e07155de94118b123950634`
- bundle hash: `sha256-aaacccb82d5f1ee28f800a30ddca9c32251032c6af48fa4d9fefc2b2578d5bd7`

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
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-cd4e50141a1c8ca414403e0e911461cbf2956ad83892052f5a578735feb0080c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4de3c73a | sha256-7a5aadc11f78f81ecd19f44cae74d37702cf25ab6483a113e820983828e67abb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4de3c73a | sha256-40c0ac00840382aedcb9ffc6e03e7e12bbae7c1352009dfb8031c75a570ac36e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-959a2db5 | sha256-a2ddea19eb70a37a2f5e3a5c44f422334b0838f4cee837e0ee9abf9cb53b55ff |
