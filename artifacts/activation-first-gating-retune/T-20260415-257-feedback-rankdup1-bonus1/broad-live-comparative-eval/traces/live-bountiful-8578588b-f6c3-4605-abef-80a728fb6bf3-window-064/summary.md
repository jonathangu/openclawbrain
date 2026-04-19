# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5976790ca942312652f7ed18bb2acfc7dc50f422300df52366315661beef5eed`
- fixture hash: `sha256-90ca4b5e31599f276df6b4ae45b8fe949a2ef12d77f2e3ecb7cb55c21378ce2c`
- score hash: `sha256-53d819b9375a3a77473a8aecbe5e904d351ab720b254842cfcdd477e6fbef6e0`
- bundle hash: `sha256-e99c2d66efce5e1ccc0773680a903f99c667c70e7f93ee981f87a4f1ee726fda`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df55308d9f9f7142399061b3ae503fa11ed1103552e4d35f047c53cb2babd5e7 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-81c2df54e0b9e99ef9bd770f9e603307e69ea632367deb696385ff1c076ed261 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a4635177bc1637e215e393bc88ee0c80199607f295d05ed7739baded7e83068f |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d349aeecd417a2fce0bf17cc73a67f4da20afbe77dd248ae9cc349077e6a9b28 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7bce2fe4 | sha256-9d70d75fb93d0dbaf0e7e18456c9a921d71cdf1df4812ec6f821a4a68225d7fe |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7bce2fe4 | sha256-3d72c4b51166176c675130d6fadd7848b713abd5ca69eb5c440a8c4696ee6ed5 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7bce2fe4 | sha256-9d70d75fb93d0dbaf0e7e18456c9a921d71cdf1df4812ec6f821a4a68225d7fe |
