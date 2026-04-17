# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24e1e9ea471d19d207e35c598683b69d84119849186e1c11e6ddd97932c4aba2`
- fixture hash: `sha256-11642bd40eb6fe8c9d53921bdb1bbcbbdf6e5f35f00a6469f30893bcfb466a96`
- score hash: `sha256-8c350273462152021018e4ada59469f09c0f493a09ef7288e81561aeebf1fef2`
- bundle hash: `sha256-9701011bb271227bfc89d494a02033929cf045bfa8f174a59c40d87c1903cf2f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe10a677dfb68dc498fb14e838ed3e08e036ad9f9df81513ada323fcaad39838 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b80e6d318066a28fa2e41f8a5b413199421a7fd33d5f101b453acf23bce70d5c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4f3278fb7f5e60a621eedd299bd4916642a69a2907a80d6849b2994c7fcea156 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6cd0f665daa655f775ea4b89f1f7793fd3a9879fbd0b5d29e55aa99da4d3b5a5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b633c3b7 | sha256-292d74e1105712bc35e9f9a6dbaed71c216a7bebe2cfa55b4941099fa5b6d96f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b633c3b7 | sha256-78431554e894c8e9135c31ee512982def035f289296e0d6efa2cce9b2ff75721 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-df94dd7a | sha256-c9cb0289211c036488c2f68ebc2064afdc1509a77f4e1c7311f3bd57af367977 |
