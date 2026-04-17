# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079`
- winner mode: `graph_prior_only`
- trace hash: `sha256-81d89529d4ba3551ffef2373c3a90591f4a3287648e2c06c75e207e29f8e1526`
- fixture hash: `sha256-ef31f66fdeb7a284c6c5e031c684ec09c55fa37e67e6013d84cbbb7caa013474`
- score hash: `sha256-503860611658b789320a4cc4aab96bd65d87d72bb61d375e69efb77f2e283e7e`
- bundle hash: `sha256-0f4a12bd79649fe9e50f1af7eb7229401fbe142ddbd2147a113ccfdd65a87c80`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b4b5bf35e2ebcc8ce20efe2342fbd2d24f5f0b713e668d8e9bc9cfb1b1256e40 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6dfe28029de6de0e0e54e467cb2c229e1f7f32c3d975f33a156363d08005a956 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8ce178d5af439da9420b23f2031c602071baca3e3b2ce5fe1f9e2d286992a2c5 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-a1d2b62b775cfa265ce81aab2265fe4b52aca2b92b72b8d4c7e1137d3d967c7d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-18a36382 | sha256-f229f0a064860691f5177e76873132d08613d20eda175a328fce988f182e3874 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-18a36382 | sha256-8a2b0ff89f94b98feb04ab4bc4cb2f32cec4b990ddcd19dc856a939898d2da14 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-96cbd8e7 | sha256-c5d7a556e72085162912c3c76bda1ecf7c1a30bcd8b866e90373607a8082c097 |
