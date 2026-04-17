# Recorded Session Replay Proof Bundle

- trace id: `live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0f24d3812e038d9d2b67d9309de9db96cd24c2faefb0b5dd93caf569b3c1d1f`
- fixture hash: `sha256-6b6c634b067ee2b84c6981ae8fc0d6c41efb6194e1723d88dd7d0087036cd1ac`
- score hash: `sha256-f163e0c6420f5499286803029b996d615e61a0864e32f4d8f23cb4290dd68c57`
- bundle hash: `sha256-86a285683bfd05dd2c547fc9e154567065f44d8029dc77b35cbc2bf2d90983e6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0ea26975a8365e08832501930a2890706222216fe363c833adbd0065a774a3f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-641d3b899a5c2cff5a1803df56589d33916edba8ec2a1e564feda142d662cf5b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a62040785efc12594af981e2d1b3b155c2b7bce60a23b22b5cfd627225f7417 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-dd138d6a6c4b924c9e57b31b80c72ffa35bf12544d80a37c6cca0e4b3df21a4e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-75e7aeac | sha256-e0fadd537422e18aa437762b25906ba69ba183e74404c5841d35a39252339c76 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-75e7aeac | sha256-5eb0ffbcabc864124faef718f427f4c47b0617b0b39120af3d93a4265b273dd8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-767b8a47 | sha256-11bf91c4e1aa31936ef94146228093f8a6f92cbdb6037413c0746d0d7aa497bf |
