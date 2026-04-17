# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3f260af2c7b68b1309e9a87df75f2e99f6d28d47bb3f82fdbd20cd787e51e3c0`
- fixture hash: `sha256-4a50ee1d4a23bf54584481d6c799516fa1f1a51aa4c19299da0f6a6b73848dff`
- score hash: `sha256-cb59ef6653ac56ce6120671784d20427f0732a5b285ced6b2a298cc99376338a`
- bundle hash: `sha256-9faeb2ff7cf1dbfd8d1cf88cc556505127d9f2496eb498e15bde0cf2ad8c1631`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9b2a597464226db9617a3470772ef24fd543ab0477b7bbc0a0ad5adf41bc0dc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-439ec3297e08f14f82450ab7ad220fbc89b0c653623479ff1458a1e6c959d843 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6978d04acfd64cbc553f1b8607101e0488ad5fe56074b77b17a3ff3862e6808e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-67e06980013ad34cae82622af32a8808207b9bedd0d3defae36e25c635ec34d0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fd2102fc | sha256-3160864039be523bb2b76e3b3020eb7adcf74b8bdbaab7a8325e0334b96ff367 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fd2102fc | sha256-297d2028d9caec9227fcc6528f83caeef1d67d4ca3091e9a0c4c4b3c69dd7439 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e4fe19b3 | sha256-4ecd43956127fa3d5ea11864d47f13933810164c3ba5aa72bb94905af607dfe2 |
