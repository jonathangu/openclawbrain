# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8d30d0b2ffefbdcd1e1a89d75d980761c51cd05c50f2c3cf1f693944186af036`
- fixture hash: `sha256-029c6b1d164f9bd1c4692f0184b6bb3b57e3ba2e59663e9c61a6962698d01e73`
- score hash: `sha256-e5c884c897d70fe50ee495347f10b7f469dc3764f24d66e13a3b3caa0770c2c1`
- bundle hash: `sha256-72ed0ffe3a091b6779007a7043f1281c865bb43876b99e0129a8881b6f489e92`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6eb030a8259079868b419f4ae1a6c389dd22240eac5e867e187ea0fab1adf6c7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a5f0a8a191675050a4382a39e1e41a2ef35fd2c234c9112ebd2faf6f81b9afa8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bac6a56221c4241050f7f4438f718ccc8204e9804f1f9a8e8042d485c2192bac |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c1c9cae4312530aee35405245c1a4b4d02ded0ca77ef41bb776f5b9d5f9808bc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bf0b8d86 | sha256-c945462e45e5599a25659d371aaa939fc8f045aefb5ae13bb4dc66b00e9f3da9 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bf0b8d86 | sha256-79f9ec9e595d5a064e6e060dca0d3b2a997d20d7d449de74333103c11980cd68 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-11f338c3 | sha256-dbdb405500a5c3a0ec9f2fa2f229cb60137bd4e7319c1963fc52653fc90be8d5 |
