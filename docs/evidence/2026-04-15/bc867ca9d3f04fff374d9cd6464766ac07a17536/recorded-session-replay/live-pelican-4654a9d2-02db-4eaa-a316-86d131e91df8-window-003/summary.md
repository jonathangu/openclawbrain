# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003`
- winner mode: `learned_route`
- trace hash: `sha256-871a8d0a2f1d4e43acf8de9d8e6956ae4d1ca9dd0a419c5265c96970bba52722`
- fixture hash: `sha256-219199343b7c6d3ad1312b7304ed4e0c3741109cf5c94240ae657c56e05e2f48`
- score hash: `sha256-9c993aac1067ca8c0fa02e5780156b4be6e5a18eb3d7c550ca4bcc424ad5377e`
- bundle hash: `sha256-784e8af10a7ac8dbd5d21e8028a7479d4c05bf4c11c816dd47a98875f8179e8b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 70 |
| 2 | vector_only | 70 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b917e276e14b121cc269645e14a5fdafe3dcdf3d48a758ee09ff6c7e3bf5cdd4 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-3bb42eeed9472f446c14f42e64872b32bd107de037a8f901886abcde8bf7405e |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a7a615a6f995b98c8586133f2fc456564901e2c85bd76a066158db472a1f3908 |
| learned_route | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 2 | sha256-26c266cf75b3b6e91367580013cd16fbc0467fbd7f8b5645d30587962b68f10c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-e33f1a24 | sha256-1aa54d3ec8848e77b1a7c1cea1df6f2f57de40820b4554467ec8e1d0659d6f36 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e33f1a24 | sha256-e7b55fb26d57cc2266fbfc05e7baf8c10d61e34698e70ce291b0e228a3ce9066 |
| learned_route | turn-1 | 70 | yes | 1/2 | no | no | pack-e33f1a24 | sha256-1aa54d3ec8848e77b1a7c1cea1df6f2f57de40820b4554467ec8e1d0659d6f36 |
