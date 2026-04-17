# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c58ed04d44aeb04071688c4a26c4c689e25ea007697f349c3e4c8fcbe3bda533`
- fixture hash: `sha256-ad70501e856aff4a57d924d7225c4dc64463e70da2f3e42777305ef85fb46a26`
- score hash: `sha256-d9e596bc0c32f0233d64e980eff81de138f1d767b42396190ea7019698200b1e`
- bundle hash: `sha256-69edd82667234da523203c3b38056a7a65147a71fc88dadf33f2dd14afd50d61`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d770aca06ab90e2e0a0ead714079ce642ffbbb18580e6acfdf4fde922a74f5a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8950a24c08ca83ccacf6b7573e5422e7d09abde4a12b05a4b627a02b3f7678db |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-169f5f1f46383da2e056e1c550151f1b01c7c67ec5745320bdf3988f119e305e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4cf77e2d8b69121184bec41e033a843c49074d34ce811c567a6ae6cb1919ace0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d9a62626 | sha256-8268e02c7c74864d15a33986eefc182fa71c9ed69f91c76141140a2648493c57 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d9a62626 | sha256-668a24465b0fa42aad90b24fe5d95551afc60ad312b2b43d9fa25ef3db7eda18 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-6ab02a03 | sha256-24c50b052cb832b7f90a92897d0a83253d36ecd70baebad5afbacd1a77b2e27c |
