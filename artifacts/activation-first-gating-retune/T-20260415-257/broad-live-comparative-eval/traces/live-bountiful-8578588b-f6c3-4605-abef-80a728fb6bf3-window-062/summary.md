# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062`
- winner mode: `graph_prior_only`
- trace hash: `sha256-95a304cc4e6fceac322fff58f14559b4669ed6895b2d6fd1036b4ee05824dbb8`
- fixture hash: `sha256-c3ee8c1fbe9d9a70f4d964351557d76b48d009945b0ecd2ef42662d9e85f4aa1`
- score hash: `sha256-8eaa7fbcb4dea84ffadf60be956cb734135a4c40b2351a531e78f8248495da4b`
- bundle hash: `sha256-03c26e5879eb070cdc7ff401719d8d941c00c878c8d44be73700f4570eba97a1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aec7ec8f6799a0ee01f6b4130aafafed76ac2a835827ba3dac19bfdac983b407 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-abd9cd8a5bb35c908066738b272303aaacf8cf7d23762755479190d78d5fced9 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6e690390e3377b30f0172681528756e38c890308ec031e296efa5c509e017af2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-4fa6d5b0b123a53510c3a88849e87ecd4540bb11f97758fa3fcf1743bcdbe79d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ab28f23b | sha256-c28a9add084f985f53ca62efc1bfd0294109fbfd8eb27315d0a80a0dc05c2122 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ab28f23b | sha256-e0f753b476b100d137399e5014db0eda03a780f7cf948406a95f3cb3098b1ad9 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8a220716 | sha256-a142ccd55cc478315d06457d1a5ee9c49722e438ab9e8aabf0925cbb598663d3 |
