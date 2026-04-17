# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ff7777be0b897266208f103a2bd1fea9aaefd91febf0ea117187545ef2d2014`
- fixture hash: `sha256-7993becb690144ea7d947bba5815a89834c7be01fb7391679807d26712c8efec`
- score hash: `sha256-051400268df1822a912d52be313fa004e626bbda9c213b3acafb53b544dc7e09`
- bundle hash: `sha256-d1e06f1073fde46c2a4bc6edf966a33833d210b44f5dcc8cfcd4e61f108af090`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3fa2d46cbebd404032c589d360c14fd0cefe70dcd15d65b4e1f8159657f983c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e8287f41e8b573df1a69a7d67498a34f99ad39ac42ebefafc88ce2649bde157a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-adeeae4e01802993b4f8950bddb0add36205b82aaf3c8ec80c4e5cbef8b0e1da |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-db800bcb58c759399e010f9b3c5031c6f55116d41a40ac8b06b7014b95fece0a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fd64cac6 | sha256-a029fa6ab603812c674c0b876e47fb75c52bff55573d896c177f34570ed6b1a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fd64cac6 | sha256-2aa7e0e4d9a5c9a29d3f0d5c7219163ca9bc95ff21323e27b092af990f81ee22 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-48c1dcb1 | sha256-c6341c4172288d0255c2179b7560a0e5189328fa7daee9d00a2999215807aff4 |
