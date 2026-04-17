# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8e3583bf5dea5a97f411db7a93b626a380f7abb1e46210bd20e8e3bd2bbad8fe`
- fixture hash: `sha256-3b9683750d86ab2808adcd8363ab4f3221db9cceab2259c7bc66ec4c98677b32`
- score hash: `sha256-86a5a3182c34b2602b5c47331d20d21e2b3072f84e9a515f63a8208e9e08881a`
- bundle hash: `sha256-18b2f1635588dd9ca17f28adb5d7dacc85c2295c462d604f0243d9f6b6a11212`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-19530c873b308f4dd0b2f26574b5efb1eeec52061eb993ae949797bdd7b8e58c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-11a908d29f872883fa3967e219758b36fb54e088dbd4de01520397ecd28ef869 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b52d85ea0c4ecaf9d2ec0ed1003fcf4d0d0db80f6df48a756e0f1f184e236414 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-423ac1ee3dddfcf6a8cb39c73e2ad7348a2ecb86b2a420d324c5224df32a32b6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a39bd689 | sha256-04183d66cbd531eaa23d24c93483f398f7b0e1905811aa849f950c5680bc7b25 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a39bd689 | sha256-5a87467dd0c9878a7f6fc59758b8891bbedbaeeb4d85c67850b2ef54e30cd18d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-4b6d18ba | sha256-ecba964f4f594822628da5a2d406d5336a59d2d1dcb401a02dbaa281a04a52b1 |
