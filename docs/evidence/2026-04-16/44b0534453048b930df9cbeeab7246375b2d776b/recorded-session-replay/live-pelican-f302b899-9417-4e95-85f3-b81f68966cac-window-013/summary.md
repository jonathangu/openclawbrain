# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-56c5cb0bbf3fd4c3b31b5c0ab401ad3e4676c774ca7f6d545e285ace8c5c1fdb`
- fixture hash: `sha256-77236387d32f039002239433f6a8c01de43cc1e1b10880d323ebd379dc420a0f`
- score hash: `sha256-0e828703ddca742ad6e2895a189cc6100194dab79fb1f40395db5360b1249a43`
- bundle hash: `sha256-c64002f21c4422665a903895b178d9a00548231d5f32bfb858b19ebeaa5a16de`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4dec047b876b4ef1cbff2ba1d3926376bc0c710b4b08c16a2a7795d5ae337d56 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d19b9fabfc8e5fd61a295ebc9c1c8c377fa1b04d268c2fb80cec77f07d9814e2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b0f9be574b3e5968336cbf956d9bc19feddf25dc9328c3ebabafcff9ab6f2938 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ddcc641c4002ebc4996b7c86888d4d72a67a465abe2341119c4cb2d200dcf0dd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-40fcaae3 | sha256-9833d9f1da4ed671657019549fc8f4f74510947f4022d9c0d56b1bf5e4bea964 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-40fcaae3 | sha256-4071a1d99d1f61165499f7accce698e0ba7cdae6c21600cb0d049ed24a7febec |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-f52d31fa | sha256-172d606f070823aa0418c27783e53b6a9478657eccaaf5dcbfd28adc3f23fecb |
