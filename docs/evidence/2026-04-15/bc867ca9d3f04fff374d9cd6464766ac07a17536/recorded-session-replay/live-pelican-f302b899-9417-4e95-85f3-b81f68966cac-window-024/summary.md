# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b224dc602b7429463a9b2fd5346afa6d3382bb3fd84bc9d3cceb0d3ff24896dc`
- fixture hash: `sha256-493fd471e0bb608979cd024ca51b9104b86ec7063e95845a4d6e7076002d21f4`
- score hash: `sha256-482188603e71965656f794a3ae1197256d6f572bb3af65e8ab7223eb6366a912`
- bundle hash: `sha256-f5fd86d2dc851052ec87f2335ac4e05ddfcd90f6e6762517ec534cad2842556b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff276f984ca7449fbf40ed52f8c73e2aedf05be900e45cdc0a8a0b8a46668591 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-89da8c2c1fb42f9d381ea6d8c77c6291d1259ec110905da5c8758e335e7d123b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-24551937965e6f772be3213fcff5a065f7af69ca0bb1c93cd0e45ea3109f9c87 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c25bbe11848a2c5d12349e14adeb427fbe431f3f29f60c65e20d4867f35db3cb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-253df84c | sha256-13d324d0c907f3d21ebd9867821f4aaab4a7aaa46506ba0d32150c289bd6e60e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-253df84c | sha256-5811eb5461d2141fb4b552cbc190825fffdc00fd8e724138acac03d84f341b3e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-253df84c | sha256-13d324d0c907f3d21ebd9867821f4aaab4a7aaa46506ba0d32150c289bd6e60e |
