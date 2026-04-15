# Recorded Session Replay Proof Bundle

- trace id: `trace-plan-regression-workflow`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d6c19595e20ace6fabb5b7541429491ac83c38d9e852f37f0b5ee0bf60e6ed38`
- fixture hash: `sha256-398d90b9686a4dc952239f413c84c29595706a3f11e555898846a0d70de73153`
- score hash: `sha256-f523ec89a8c02c755e2c80823530274a74179d47f768d4edccb89c3b145a2570`
- bundle hash: `sha256-7d7f09e4cb648c89dc31ce7eecc203b69a276716366567a59835b8153c788181`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 88 |
| 2 | learned_route | 88 |
| 3 | vector_only | 88 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 12/20
- phrase hit rate: 0.6

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 0.8 | 0 | 1 |
| graph_prior_only | 2 | 1 | 0.8 | 0 | 1 |
| learned_route | 2 | 1 | 0.8 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/5 | 0 | 0 | 2 | 1 | 0 | sha256-90328e9b0c4f9b38528d7a84c935133ee1cb08f73bd2c39b66d2b7ab6828c750 |
| vector_only | 2 | 2 | 4/5 | 0 | 0 | 2 | 1 | 0 | sha256-9f753531f74de7da80060350c9810a057fcf7fa7bc090033e25c2e5d586df035 |
| graph_prior_only | 2 | 2 | 4/5 | 0 | 0 | 2 | 1 | 0 | sha256-20cfdb2124b87dd7dbe58f03e81d9a9d8a4af7fc04b6050ca267fc729810333d |
| learned_route | 2 | 2 | 4/5 | 1 | 1 | 2 | 1 | 0 | sha256-49a9dce374e6c7ad3c333eaecbf965be031eb588ba343efd7f66120f2064039d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | regression-workflow-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | regression-workflow-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | regression-workflow-turn-1 | 100 | yes | 2/2 | no | no | pack-288281bd | sha256-ed38b0f3c752b0c55bddb035a2512944255a50993f9396d393d145c8af26d907 |
| vector_only | regression-workflow-turn-2 | 80 | yes | 2/3 | no | no | pack-288281bd | sha256-fe035e7f61174668e6ae735f3dae9f60e47110e788df31ad98d00384471bfa1d |
| graph_prior_only | regression-workflow-turn-1 | 100 | yes | 2/2 | no | no | pack-288281bd | sha256-ed38b0f3c752b0c55bddb035a2512944255a50993f9396d393d145c8af26d907 |
| graph_prior_only | regression-workflow-turn-2 | 80 | yes | 2/3 | no | no | pack-288281bd | sha256-fe035e7f61174668e6ae735f3dae9f60e47110e788df31ad98d00384471bfa1d |
| learned_route | regression-workflow-turn-1 | 100 | yes | 2/2 | no | yes | pack-288281bd | sha256-ed38b0f3c752b0c55bddb035a2512944255a50993f9396d393d145c8af26d907 |
| learned_route | regression-workflow-turn-2 | 80 | yes | 2/3 | yes | no | pack-167c08b4 | sha256-697339927a48772f20a6b7c748f75b5b8cfa82466abc14b47b3465ae7482f995 |
