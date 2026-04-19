# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053`
- winner mode: `graph_prior_only`
- trace hash: `sha256-90fa2d4d4139a1ebe44459c22c8727829266007877108f45fae96bd38c29ee19`
- fixture hash: `sha256-cdac1a3c1c52fe7f99167e3c99b2296e6c3d58fa57db9f5982bd144cf8ae1b02`
- score hash: `sha256-bd1aa67a14fb6aa10de1c2127d637e7c89cffbacd52ff13f5c4fac3509978bba`
- bundle hash: `sha256-9a29cb519cfe5083652330c96245d341604da2b033140b454dda73dac7d6c8a0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-24482bbabb11a52346eb943534814017cb04b7bc645577c10d973e3de61757df |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-9b79819751997d9ebc799b442f40354f798ea200c63b36f2f357bbd772f878b9 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a142d0d37a8cbe79a9d754d28d659994d4c2b07910dcd4cc51edbe73af0ef4f2 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-cf5b5a7ffc2b9ab99daadb880d6063b3c243c8d596cf7d74ebd43111f9d7223d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-67d14db1 | sha256-dc952a975143af85767ae06899018d4d3495c8f52e987c156c9923eb7933d735 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-67d14db1 | sha256-c05c2023ea62c1276b83e051b6a350e67b350c03b5b077292a21c4146f5badca |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-67d14db1 | sha256-dc952a975143af85767ae06899018d4d3495c8f52e987c156c9923eb7933d735 |
