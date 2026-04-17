# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bbdcea0bcebe80c4396f90ecefc21d80712c695057e642f25e10243f87c4c7f`
- fixture hash: `sha256-e1e5a273f33109e97564303fef433c5ce8b0488cb943d73553c438e3bc4b82f9`
- score hash: `sha256-702870c87d87f02d1f93fa291ce5af6870a9d551e67351a7c67579ffad1d529d`
- bundle hash: `sha256-de9a9b671884865d900f780848205eba608c8f636d4b9d6ec043d1bcad39b314`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7d1f1cbb97b76e814b60c9940bdeb937cd1496e6d58b8c2941e362dd90a4031 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a949ab57d65c8d6da3a634f11fe48b2545b9d184fcb3511445ebc23523fd355f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2518bd6b9c1530223d99ac7aaa43979333410fe6f1a2f6502790e77a40f09e85 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-90ab6ad706c83a6d14b60521486b737b6f32ef865d5d5808c1ae7f3c91be1b7f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-51f16d2b | sha256-d109ab8b6187079a02d99c109445c1737a6d37cc36658f1ba3f0f8f532d9db81 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-51f16d2b | sha256-c8b572937305274399fd4032617f7e62d1859553d5c62890c5627c9b2bda6074 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-1c5b9e08 | sha256-f64a9243b3a5db21eb83d3a41408c04736c913bce7dccd4f75572c7947aad541 |
