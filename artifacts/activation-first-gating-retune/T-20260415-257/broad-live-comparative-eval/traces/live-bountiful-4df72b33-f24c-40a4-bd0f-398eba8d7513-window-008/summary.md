# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70911ec1e7805ccec970087d6c2246db12da18117e08ef4135c17a78ab963e90`
- fixture hash: `sha256-5f3ce437bc5a34220be72a905054c7058ccdfb9aee9afb407a944b39db8e43dd`
- score hash: `sha256-cb1f7701b628731b398a3b2cfbe78ee464cc15c301bf6493c4040f087b556434`
- bundle hash: `sha256-3a29e375990820321b0b02e138655b0e6d3b8f5b20de5c196eba7dfa5523d263`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c5564d3b460f1097de136a88547e9b2bb9e15503e1a0ceed301551bb8e7b5353 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-4abcaa9c9de165c85dd7ee3accbbd46527213913b90726a5c87209474ad8f630 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-5e7abcb77aef1e0b6861e439423718f2641d57c55a8f5b120252d13a6aea0f7e |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f9bc930ab6dc918ef6b84e2507f6d4b1c56b4b8e9654691850fd902e6e5f4f23 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-60e8e66e | sha256-faf69ef94a9bdb1ce7901afd2a9be13c3c53089642f56c3ebfdd02f2f619fa78 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-60e8e66e | sha256-8c09b870e1dd891134c5fd9680d515d6bb62389fbbc658d88d8d624a2065056e |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-60e8e66e | sha256-faf69ef94a9bdb1ce7901afd2a9be13c3c53089642f56c3ebfdd02f2f619fa78 |
