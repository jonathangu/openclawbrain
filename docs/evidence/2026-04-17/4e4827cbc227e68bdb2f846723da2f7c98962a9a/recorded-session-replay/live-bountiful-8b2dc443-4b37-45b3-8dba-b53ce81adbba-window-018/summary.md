# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d5c4c3a45a443f8773663b6b718260034f858fe43001100c3e44deaa92dae64`
- fixture hash: `sha256-8ee17d6b70fb97105471476aa616629c3b433fcacd6e10fa09857f62252427e6`
- score hash: `sha256-f098b79a6fcd9b0835df0ae945da810d532530d984654090a01575fb5a1d0905`
- bundle hash: `sha256-98063cec84806b5a63694e821fe50ad852f956d6d2eff1eceda31f69aed03cd5`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-09e55df402125c6b04d503b2df670ff995850f1c31d072adf7d8fb44788c9b43 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d0c162a791fd2e884c02aa35ccb4948ba40fc6e885cf0fe7322b60ed92666636 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5e0b52b05ee4b7f21615cfce92796051b5d46b567037cb84e861ef969ff2aa99 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-456c236ce989858bb877cd15df47d638e8cde0e853a7fb0f1d960b7d50e066cb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b57af8a4 | sha256-2e9d90286c2fc65ce43a68b35c6dccaf7a8664350780845ed3bb94e3089b6541 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b57af8a4 | sha256-d4c5684b2902764e0991f928505c8f0c93d33c459c07babc89d7b42dd3d518a7 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-30e0def9 | sha256-f42935f1d1d616583b8e1a571ae93d5e5d121a2868a7430ef75d88e80e7b6e05 |
