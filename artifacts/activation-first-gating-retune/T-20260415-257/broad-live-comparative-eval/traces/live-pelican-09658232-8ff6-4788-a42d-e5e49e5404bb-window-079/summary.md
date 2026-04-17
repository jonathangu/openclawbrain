# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079`
- winner mode: `graph_prior_only`
- trace hash: `sha256-81d89529d4ba3551ffef2373c3a90591f4a3287648e2c06c75e207e29f8e1526`
- fixture hash: `sha256-ef31f66fdeb7a284c6c5e031c684ec09c55fa37e67e6013d84cbbb7caa013474`
- score hash: `sha256-16dfee5823308e1616a313f4574eafa5cc1c730fd30f142af57d9c614b173aa5`
- bundle hash: `sha256-cfddec06abc03874a074abf0a9800c755ef12a552d50bccbd6fd077ec6c20cde`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6c3d0a21c813965365cd49e0143e50528639558e4c750cbd07465d6f285e9062 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-82b569f73456b860933563a829440d1661c0a370d15983142393c10776e5701b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b51b6514cf15f9272bf46c9ff32f682c653ec1053246a7f3eda80b05864ae29c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e2652140 | sha256-23108494a76336829b672195ac85590736fd319eb8e20c18ab0bac0711ce8807 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e2652140 | sha256-a81ccdb3831238b3abb23b8db8398cec6a5f58c28ee8b00fc1cc724e71a5bbe2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-608d96a5 | sha256-8b6ad2e1e90b402126c230d4c5891f5b615e750e5ce012ab971010b496efd83c |
