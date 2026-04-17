# Recorded Session Replay Proof Bundle

- trace id: `live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e725d27535b9af2607edfc55a55297b3d0ba2750ec3f143ed7ff11bf77f51432`
- fixture hash: `sha256-ad7016b7c1b3014d2314a0a7c149b979d45fc36075f6fdac2c19edc868ece1f9`
- score hash: `sha256-d634de1a85c5251886f631e6361a9ab5fbff9a4b722b877780b3bcf4d0730238`
- bundle hash: `sha256-48bac3ad75d1c30e142c91a6db80a4944e73391419e7ad8af9348887ac0fee08`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-876d8e0d55a1dbf8e84f9354102b290654bd4fd9a97c3f26b37887de51274c7c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4ff40eb8c12065e730ea21af3be80b45f511979c2ad430133a4b8ac9b752c7de |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c18685919515ac54d0636d162fc3f3399281a34e33436cd94c316b26aaa806d7 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ec3d0f2a9d68503892d305ccac333f74e565f5f1161ea9ba4d863c82dc4533df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-565bc820 | sha256-c3d14f90e270abd817efe6222b274529fa3e8ee67e2dd7de8e1dfd4a91ece76a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-565bc820 | sha256-91957ea10d0ed40d476ec037aac101bd8dcd47b7cbfe4d76ab7e89e27b2f35a9 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-c1f59bf5 | sha256-4982315634f620f386e62aedf5604aa9871361a32e2f694a6ba8fe9095be7291 |
