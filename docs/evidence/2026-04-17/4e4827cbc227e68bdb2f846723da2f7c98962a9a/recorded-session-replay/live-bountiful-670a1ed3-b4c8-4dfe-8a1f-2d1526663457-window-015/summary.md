# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06fea57a3516f2a337e636c80dde5aa0f7b5c4e7b115ef7c15ef4879727a06c9`
- fixture hash: `sha256-0bd1e90ada8a113768901038367ef3359fd513f44e7b3d01e72effd5c2301b57`
- score hash: `sha256-e7a8f6c6cca9daa6f6e5563de98edc8446353de5c0206eabf705daaf2121a7f3`
- bundle hash: `sha256-423f8a53c36ef78c99f44a1f3578bfc0fab9182ee997806b7f9225806cbfbc5f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5c5ee087496e1c83dd50cdb77e530bafbdd0a3348e86d19deb3da1e266821f9a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9a5dc4394f196c6ce2f5a9d6361e37a3b93be5012d162d44cd04777795e3be2b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-061aa1cff3e0b857583eee01907239766c1c91a5d2e5c576eef6499485350653 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5d735b40b2576023853c33cfcb784ee094a37fec6a0f84691d98f1748159da23 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5478a29a | sha256-fc85ae10cb9496d491c27177d4938e619057ebb9eb212090af8f36cd5c5edaa0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5478a29a | sha256-43ce053f9e34281e844d8e665308a8a636c19d03de10b6e5da8befb9b90d0574 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ecde9ca3 | sha256-d3c052f1b903bf272adabd245c2fa460fd4691023798ec4cea0048f35d3feb38 |
