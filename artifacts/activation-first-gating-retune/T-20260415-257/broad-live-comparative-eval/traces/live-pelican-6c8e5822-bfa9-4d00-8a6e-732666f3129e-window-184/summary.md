# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d479c91d117044f49ae49da499694fdbe9a9bce3b101e2f906d0092b46536940`
- fixture hash: `sha256-afb53fed27fe0fd6a6ad4e067cb4e140573e8cbd954bfddd658b0c3c6c424a0e`
- score hash: `sha256-349a5b0ce0e493bb689be0e1b78d218910391136f9f8e4bf079cda9be37f1b34`
- bundle hash: `sha256-48793425eb4d347d9c953b48dc65bc24b9cbf5b09102de10b4aa261ec671e27b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2f4b660a3e5d5f4a920994b92d0eae72726c74b613f3139fcacbac22692626d |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-3d845d9a29baaa927502888d4be7855c32e44c60b23a252d3b4b5bff0c8d8fea |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-498e490ef24be49e758d4d144b41b1b81fa588b3983865fa98a8d281b75d5347 |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-7ab447e28d257b04929fa4905ac56b313f7b455c38f5cc70edd8bce9179011bf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-8e3e3a03 | sha256-ba407e73ce7ff31a67e931cbfd0737f024cec4589f2e6e5173a29d2d125edcc4 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-8e3e3a03 | sha256-d57ea609b92761de7c4ffd00536223ca4e3d234bf3e3919064b6ae2c0ea37cf8 |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-d3e90f46 | sha256-008ef187431b5e4c643cf2277e0421664c8757529280fadb90a6ab4931edc92b |
