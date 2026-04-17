# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06fea57a3516f2a337e636c80dde5aa0f7b5c4e7b115ef7c15ef4879727a06c9`
- fixture hash: `sha256-0bd1e90ada8a113768901038367ef3359fd513f44e7b3d01e72effd5c2301b57`
- score hash: `sha256-a64ee88f8b58453c931321394bfa1f618828fb55fa4c71be4645eb1c4e9986ed`
- bundle hash: `sha256-b86b48fdfc95b8a8c0575cefec70fb8d0ecbf88046daa8e9c542f534504bbec0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5c5ee087496e1c83dd50cdb77e530bafbdd0a3348e86d19deb3da1e266821f9a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ef7187632086f2a56c16cbde666cd67843a010480acc1a41e762798ac3f5c9ad |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c3a8a41fce43c55dce4f87b8edb6de02e1ff3404655959dbbd2d4b7704d6a1b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a6034777ecaf2d903e8b9917426ac0e5e89c1a7d96ddfccd9be0fe95819ca430 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a1ecfa07 | sha256-d1aac5b623b27d9a2d62f520beaad9f016755c0ce70661e8dfb1ab653f0ba654 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a1ecfa07 | sha256-7147d71a686b5fc6bbeab9cf4f3bf25689b06a79e35f891b9254f41f25299039 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3a52f410 | sha256-d79c9c22885d8aaf8cb242dd43e690fce875bda375c4ec5cbc5f4bcaba7549ad |
