# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fc1409b104d617856751474f01593056b66d1b2ca492e8f5dd879839efd10f66`
- fixture hash: `sha256-8310747322d42de0fb2d06597a429aa5eb75a2026f88cf3e458dadef80911084`
- score hash: `sha256-bf1f1b1a43d9b6eb65166e46630a89268e3b4cdaa006999b9a45b2f9776f36dd`
- bundle hash: `sha256-654014a65338f6c2505e99e2dc1fed320678aac27738e388ea1498b9d866c31d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0fcb97b42b2b441ec8190e1bb06fb82b8bdd1457d8fd6d8d105b2684066c5870 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0eee5372df0f47ebca9f6d1499d37796b3aeef10bce21b6021877aa72e0599ab |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f5beb04a36757e2895e56b086eee810df6cbbf72368d3d94b87e23a81f8105fe |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-64edd4b1b62cd3d1b2de9ded24cb9e24aa8491254529f8fb7a6dd93a6d6b948a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-033d69df | sha256-d288f074f5920f077fdb845868d1b80b8852c3bdc0e1e4134e10bb1e7d8b1166 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-033d69df | sha256-149d6f8a7718bd06ebdd43451c4d0a0be5afff851e259bafc2eb3e020841f49a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a63e49f0 | sha256-9b308c7dae7d751ec2e91e5441711a2351481c603ebc778dd9ea776653431b58 |
