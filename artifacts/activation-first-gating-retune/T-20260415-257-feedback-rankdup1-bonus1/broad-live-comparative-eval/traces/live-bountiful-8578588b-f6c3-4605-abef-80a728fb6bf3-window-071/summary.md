# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3081fc6cd2140702648472a4581d5d967423fa00a434b9483a180d3d9c6b8fc1`
- fixture hash: `sha256-b5ac8e7827ed4f12c1ba7efd325a7c14f155c3ac4043229edd168af627bd6e56`
- score hash: `sha256-169f11bdf7477fab35bdf5cff9d209e8ba40a44b59666fb686012182a5a52933`
- bundle hash: `sha256-5782a1cabfedc9f0fc791e4111caa79726476b614b9c1c7959b59a908e8f476d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7d107ec56aa4c4574e497f7d1f6692d0d1c30fec9266f7ec344bc3b377fad16 |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-bbddeb9841ad3938e592afd9f53b8531bdfd1cceb2f9ffb1df204bcb7cae7d64 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-7e212e4959d665fd1bf2516bfae3f9fa57e6848b969a6e314467743e6f815a80 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-dff8ebe4e1a72189ad0147027a41c33a64f0904d16aa4922509bbe1ca98b39d4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-fbdf5392 | sha256-40e28cd2852ac7c44b29916e4ff56f014d91ccf9e1e47548458a3068f9332685 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-fbdf5392 | sha256-40e28cd2852ac7c44b29916e4ff56f014d91ccf9e1e47548458a3068f9332685 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-fbdf5392 | sha256-40e28cd2852ac7c44b29916e4ff56f014d91ccf9e1e47548458a3068f9332685 |
