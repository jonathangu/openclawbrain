# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bb64d18c7380c29a36adda7f18b9d94028bd2ec79c3f043249c311ff96079b77`
- fixture hash: `sha256-4d9c945d16c80ffc64625c9921a10c1aa73d0e2d0d7dc96750c287fa87ef0a3c`
- score hash: `sha256-6b911c9784843c00f864d2d19e6e5a294ada6e1471788a6e5e63fd2ead4ddc63`
- bundle hash: `sha256-4217b1ad1c9c41a894377a1b72adba94636f4e6e4927039bbb99046b549ad64b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2057eeb74350d5472d7a207a6cd23d83fdfc1cbff7a9da70502d2c9709cf85fb |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-77b87998589cd296c7785d5661021d891eb94b1ddd659e0f46d28ece954f46a2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-86d6bfcfa83cc3559af5778bf95808595fdd742151ca859fb83b4d5cc4dd54c1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-9fd707b646b6aed15c68101d3ebb88826782bdf4a301e5233ba21db5b8f3c504 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fcb0577b | sha256-2be1aefd011ff1eda60e74ab63c453639da94bdfb4018af4bc717ad80b77d85a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fcb0577b | sha256-d13bb50f90355ff0e95280837626436139a87efead14dfa4ee84f66126942740 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c27f133a | sha256-1761a142141301d8ec6d91e92972c11b8393a9bebb4669671bd76c318ed4452d |
