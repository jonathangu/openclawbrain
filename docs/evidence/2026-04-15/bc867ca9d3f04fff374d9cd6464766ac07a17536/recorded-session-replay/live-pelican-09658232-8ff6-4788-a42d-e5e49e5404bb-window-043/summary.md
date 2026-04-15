# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5dd6be200875e55287b6b027aec6adb1211d2b987b83c6a37b985516f7118529`
- fixture hash: `sha256-eec602e66445ff4dd47c7240e799fd3d8564ee87f3fa97f5e6b5673abf356c14`
- score hash: `sha256-7aee5c71f280aff6bb7451974017f2cecc115be7980c015aeda36c6d0db1ad16`
- bundle hash: `sha256-48aa7c8faba247a9d0c6701a4b4e484b1d89f401fcbb0d69043673c71a9b8edf`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d99ddcb432bf41d8fa10f8ab6904c40f835adbab6565ab293b9f4c7f5ab02130 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1d1b80917f08ed0009b8f20f3b44a78c835f29ca8e7c946183cf7f2a31235726 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-39dffce40d7a07a4c4089b01866d6aad1d67254d2878eaefd457679d0fff6611 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d7d076b30852af570062167461484a391445bc077b48adde3b32dcf5331d674a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1ef20517 | sha256-f58d1e74a3eea333a72b2fad4a12a6e52bf54a232df796905bf38aed2935b37b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1ef20517 | sha256-5158716697f0fab41f727a8895fef26f821e0e52c86a0964dcfb289b2ab12756 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1ef20517 | sha256-f58d1e74a3eea333a72b2fad4a12a6e52bf54a232df796905bf38aed2935b37b |
