# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-850bdaf9d44fc5882480e1fdabd688dfe420007059248f68b1bfcdb177c8d991`
- fixture hash: `sha256-01700f6ae7fa9661baee2d1698232fbeb6cde54e151f8324fe1800456806d50b`
- score hash: `sha256-3bdf3c2cfd1bfa80bfdc8393a761a3f3eff83da00aad4fee2f4c5da08d526379`
- bundle hash: `sha256-3fa4d663fa08b9fec0cd488b4d8d9cf9de457b01688ed8ed3a698b779ce17941`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-db5ed18ce329dbd2d8fbf4381eae760a575d1622b5dffa25a4d7dabdc4b4d367 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a4f9bc9c3290b25000022b84cb8f26d72bf2bcc7040e29bbb25bc5f955a5faf6 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e9d00119222cfe2cbd7af9b43ee3cb8307f76919cc5f516f35571cd90b33e2c4 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-ac8b2ea3addf7c7111fc35abbaca2ba3ded93b38e86c9069a36e1c9ca3043300 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-348f93c4 | sha256-50e937631f19bfbd72d35fe12418c5e1f26b30be00228dbaa49f3116322163c4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-348f93c4 | sha256-5775d5d8b3c13b8474ff5b7def4a8ff768e036d22d470d22f3d6b5a11ee7cb95 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-c88b32df | sha256-cf22e2686cd289be2015e3ca430ad23cd7e48d802586d6824b060c7fde7836ac |
