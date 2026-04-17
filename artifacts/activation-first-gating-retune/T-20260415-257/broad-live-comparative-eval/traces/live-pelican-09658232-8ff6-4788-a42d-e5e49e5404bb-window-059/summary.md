# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c32e2ebb8b7b9b5fb8d50fc85f34ad187e87c05bcad5baa201a1c538d7a405ab`
- fixture hash: `sha256-e23c11c000ae3f195bff5e2ea98696c33b399e18fdc28541e3c50f4b667d3e58`
- score hash: `sha256-5ed8be6a56943dae8c80f4b5703dc491d4fa268462f8fa6ef6bda330a4a56e28`
- bundle hash: `sha256-f31f72cb96c4445dbd3975ee77908c042a69fcb16e95322fdb69f0b062c665f4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3b40ea9a22b9a22398f998c23b04d742ab7923c5900fe44bcea6dd68bb464780 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-178cc072baa25237daff98cb3cc9dfa69e3437c457a0aa260b1adf8a4383283a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9cf215a8b95a684c154ab47d0fbe5c230e7ab085c48620bfd94fac956d6f5c32 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1b4accf44d3e62f913ea11a3d68bc469482c8eb0a2402c6dc6cb6357e5c0f302 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33340329 | sha256-5427571069179f56a7412d1dfc932e257e252f960a0e84536fe6e6795695a9d3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33340329 | sha256-fb39434336ddf67645ede54e647c15c139a86002a028682f6bac57467669fcf8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0255ace4 | sha256-33cbad6896f5cc049f15f4250abc9971dc1ed7c8248251aca656f162586e459c |
