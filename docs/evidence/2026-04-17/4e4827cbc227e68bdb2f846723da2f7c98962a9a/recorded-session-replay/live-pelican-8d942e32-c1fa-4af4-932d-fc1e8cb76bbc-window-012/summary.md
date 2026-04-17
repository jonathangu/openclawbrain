# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-874a83098560adaa94c38c7c63cbf4c86efe4c86090d606bbfa34849e336a8c9`
- fixture hash: `sha256-b06776d862580d01d558132918aaffc22b9130c1387f99ca2438e1c6cbf7e22c`
- score hash: `sha256-c73a882cc92ace026be92f134298644e30074f2579d20e4f0d012c6cf0083c3c`
- bundle hash: `sha256-09c1d2aa22a3d9a860b282d61c731ab3df1995be6227961afe84838cbf72ba11`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-74c569833e5bf27fad2f2f842fa8eaa7d60bb320f690bc493bbf6c394f309f6d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-36ec9ba67cfef58a11d5dd39d3a1cdc4e68a953eb729c0e246847760a7bc6b7f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a8f6eebaa0c7c1e3c13c453f695014e1e0fcb6db2318f1548fb8b19f84f0b47 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6d733f111c54aba35e5be50f2f6b21c00a9131e350ebf9d805fed39905deeab4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0d8d922f | sha256-27531e384cc888a4ab05c67fafbf33081b9f3c53ddd88eddfe5ea31482223b82 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0d8d922f | sha256-7f856bf50b1c4533f54db1e8390c9523ee4af60e57b84fd91e8181d47e6e36bd |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-663f0230 | sha256-8471a7aaf340f647875af60a4e7fe3433e290a09f705ed00470ad4b4c73682a1 |
