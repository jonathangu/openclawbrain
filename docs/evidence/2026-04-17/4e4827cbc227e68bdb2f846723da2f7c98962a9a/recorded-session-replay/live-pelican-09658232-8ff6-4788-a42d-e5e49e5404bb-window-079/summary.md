# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079`
- winner mode: `graph_prior_only`
- trace hash: `sha256-81d89529d4ba3551ffef2373c3a90591f4a3287648e2c06c75e207e29f8e1526`
- fixture hash: `sha256-ef31f66fdeb7a284c6c5e031c684ec09c55fa37e67e6013d84cbbb7caa013474`
- score hash: `sha256-d740ee339a0edafa9ea422c63b9281dab7e5269949a317631fe04a98a41249eb`
- bundle hash: `sha256-1dab3eae893bd83f76a9b5cbab19caccb04fa554ebb6ef6b7994b8c2ea254be2`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-61801c90bc880529512009d6eade8665f91f8d631c30a5200e3788c30ad38cb8 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16d4af26a6e3f3ca54e50c1af8bb39f30475efa490bf292c2d222dc0f5b2d05a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7527e9432c9aa2a7da3ec9b12bed661ffa1060fdd9862b28be085fe053f2ced1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0b5a1f17 | sha256-015d489a933efae7db7f9c2330e897b4191403594d3d88baad3d1c0ea0bc959a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0b5a1f17 | sha256-12fc6a17004faac92f1aa46070f24686166443b70bccd65e1841fc7a2e26d523 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8982947c | sha256-9b65fa66c8d04a326efe9a3dd8f4cca4e402a90d82dc187f39ab04468da6a74f |
