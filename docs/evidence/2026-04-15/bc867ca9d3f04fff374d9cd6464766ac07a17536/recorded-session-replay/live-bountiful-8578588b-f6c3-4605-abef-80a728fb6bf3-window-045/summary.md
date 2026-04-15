# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8e3583bf5dea5a97f411db7a93b626a380f7abb1e46210bd20e8e3bd2bbad8fe`
- fixture hash: `sha256-3b9683750d86ab2808adcd8363ab4f3221db9cceab2259c7bc66ec4c98677b32`
- score hash: `sha256-bab5aa0dce7de6a4e3cf500a160ca3113eda72875989ae7ca2e31c4e91244956`
- bundle hash: `sha256-fced42a8bea17cbf2a8513f9350f013c31d2cc245e44b22291541959a8055760`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-19530c873b308f4dd0b2f26574b5efb1eeec52061eb993ae949797bdd7b8e58c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-11a908d29f872883fa3967e219758b36fb54e088dbd4de01520397ecd28ef869 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b52d85ea0c4ecaf9d2ec0ed1003fcf4d0d0db80f6df48a756e0f1f184e236414 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ee95ffde61287a7b3eda8c7e088cb27fa5bccc555e954f4413e9441135b5f7ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a39bd689 | sha256-04183d66cbd531eaa23d24c93483f398f7b0e1905811aa849f950c5680bc7b25 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a39bd689 | sha256-5a87467dd0c9878a7f6fc59758b8891bbedbaeeb4d85c67850b2ef54e30cd18d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a39bd689 | sha256-04183d66cbd531eaa23d24c93483f398f7b0e1905811aa849f950c5680bc7b25 |
