# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b9882f49bf30cb6d948087b310dd1f1c8c43cb51ebde7842866360d6db046b12`
- fixture hash: `sha256-371ddc3cfed0332b92f92e9c2b214fd34bd05f438837cc6562acfdd4c1e2c749`
- score hash: `sha256-f851fdd8c096df4b0b7a64023f8091eb5def15d9355dbac467142527a945c016`
- bundle hash: `sha256-0a05e3d44fe7215ef1889b9fd07d1165bf964a94204076d27c1c9592acce6a5c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d3958fc805f3776c38e6e687c85563bf09e68cc8dca03392a973d72cef995c7a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-224313a2252b22d7d01811add5a5678cfa3cb8a4e8b78cd6eef8662b260a6afa |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-62b3e87b97aea6efbfd1edb2b70c119f952b1e8e7dc5d3241563311fc8a3a0d6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-53c5ae837f326d92bd8cae3745af0047e6b6de395be097f39414257adab8b703 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ebdee63e | sha256-17c901603a6571c85fe4b7ec3b598b6f64855325d416c90ccb7ddcc9bfb86c2c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ebdee63e | sha256-5796047cebaa943930bfe3d7fd498a89d3cb4db2c635c8e5c573ac265be5fda7 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2c074977 | sha256-859f099c787d6aac7b00a7e18a275c506563bc031adae2b775ec226408581cdf |
