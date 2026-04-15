# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cc22db3aaa15315761f798aaec1df1acf278bfe86338b981b1d314f80e60f459`
- fixture hash: `sha256-6250124575745297903131838786e09bce6bd0b2285afd782515714f7d74a408`
- score hash: `sha256-96d3bb2f05e42a75210af18788324f4edf5fa361065a8ee26a3ee9f766933807`
- bundle hash: `sha256-6c26acefef29948a3be438feb14f6d45b575f1a4708bd2a9a9a3fec6b057440c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-363a0ce15ddf219a167f700ba2217552de3446a99d080128478494cb795b929d |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-32c6fcef87bafe6292aec6218bc9853a4619c822c3f2ef795d0c6a42447e70d0 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-0101c9a125a4ab52d0a9099d6e71819a79f7180dbc2e437d49fc2f5bac47b310 |
| learned_route | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 2 | sha256-177e4ce331fd0b777820860a42a7742828a47373543291715fb5f6d0deb4e9aa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-145fee98 | sha256-b5bbf9cfacfdfa8f67433c825512bc186865c609d62e9c7deb425e4e1510bb38 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-145fee98 | sha256-44505243c2e7a559ff2136fccb0c0d3a28a5670c5af372eeb86afc19d9ea8509 |
| learned_route | turn-1 | 70 | yes | 1/2 | no | no | pack-145fee98 | sha256-b5bbf9cfacfdfa8f67433c825512bc186865c609d62e9c7deb425e4e1510bb38 |
