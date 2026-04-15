# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049`
- winner mode: `graph_prior_only`
- trace hash: `sha256-10f30fda1583220ffcb0e13cb73de4976d5f3f5f0f058e8e816ab9eaaeb4bc0c`
- fixture hash: `sha256-81aecda5857d0ab09faf0a56bf49fbe289e64582b0578df3f1535d5bf05ea11e`
- score hash: `sha256-9bfc945359f8be4e4b1c1b498faab64e5dd3eb013ee46a876d2ac1a8bebce3c5`
- bundle hash: `sha256-364296ea1b7342c6d92e354b7582306a9c72574d9a131dbdf7f2ca54ed8b70a3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd49736decd703dddc6036cdf0bf744059f6270cb8728fb209a65a281dd21058 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b41edfc99a6effb688ec58e6e55b97e4afaf359ec8d64762b2d8d8a4c265ddfc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c06d874fc87f470e28131549e17e47156060583e7294c4bf9470bda0215a733 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c1751a5bdbdaab2e872875f55e6a32c5661e5de6c954e1cd45329a15c97f1987 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f8c090c7 | sha256-47aa8d5e1c50ba1e2493c0c74568475d9c180d1d3638bf77497e2880cf6437de |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f8c090c7 | sha256-3d58aaa6a4abc8dfb346000195b9e0a24e09c0c534b784f6cb0b45a2d2d68934 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f8c090c7 | sha256-47aa8d5e1c50ba1e2493c0c74568475d9c180d1d3638bf77497e2880cf6437de |
