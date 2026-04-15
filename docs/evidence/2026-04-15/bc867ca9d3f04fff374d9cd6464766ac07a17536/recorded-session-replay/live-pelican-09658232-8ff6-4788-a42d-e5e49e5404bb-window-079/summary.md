# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079`
- winner mode: `graph_prior_only`
- trace hash: `sha256-81d89529d4ba3551ffef2373c3a90591f4a3287648e2c06c75e207e29f8e1526`
- fixture hash: `sha256-ef31f66fdeb7a284c6c5e031c684ec09c55fa37e67e6013d84cbbb7caa013474`
- score hash: `sha256-3cc54e53ee02b38175a80dc18781d6ca367b2d5546b3aff0217f4d163bf34ae0`
- bundle hash: `sha256-b9ccda93b53251c94461a5fcf555e45411334f8276779e93f8a20d1f8df90324`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b4b5bf35e2ebcc8ce20efe2342fbd2d24f5f0b713e668d8e9bc9cfb1b1256e40 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-82786ed4c5893593ac77f0ce81e9208a51e9a2b63014af42430d2c5a2c3bef19 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fa179f122725adf7fe457f76c8a2011484c53b92870abfd0bd79892ec540acc0 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-50dcc0e14952e4e02fb167c0612a0908c5effc0ba0c29150502ba21bdb632e61 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-64161b35 | sha256-6d3ebadab6216e46bfc0a5f7388fd4c30f7ab4ad442386a57633470aaecc318f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-64161b35 | sha256-a5215239f9cd47d06de378019701f48022759fc349c1dd143a88488d1350f8aa |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-64161b35 | sha256-6d3ebadab6216e46bfc0a5f7388fd4c30f7ab4ad442386a57633470aaecc318f |
