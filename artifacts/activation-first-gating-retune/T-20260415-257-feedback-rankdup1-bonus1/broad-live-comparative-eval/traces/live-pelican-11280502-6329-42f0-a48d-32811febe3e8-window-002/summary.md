# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e4ef5844fa79e43c9383933848095b91a4b282e9ee457dbc9f1c9f66329542dd`
- fixture hash: `sha256-7a05036812c8c043bba376d7dabd598905517ac4fafe99540ddae7c177988a91`
- score hash: `sha256-a241081288fda7419c4ebfdfeb31ab5fff5d7f4cca0ea8c81e996aa1b05697c8`
- bundle hash: `sha256-d6ee32d8c67b71bd590bd908fcc108213e29f41952b9fa54aa78c089f585779f`

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
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8d02dccc8f2c581b6afb2eeb0827f1dbd7b9a4cbd13481c6743fbe222b13d1cc |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-370b401e89662ea559e1e622c4162e2c74a005fb8649b53458d54424bc12163d |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-34bd2a08a78b263ef10e11f266c4613015664a22f4c2ea4292e8564fa120c000 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-3705fbcec466b0f396947180d358e8cf9ad205d068c3bcc3b2ce9f03b35a4798 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-7c5d5fba | sha256-326a0c352fcf1393bf40144ae8907641518b75b94b25c73a8c7e73b50f21b899 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-7c5d5fba | sha256-326a0c352fcf1393bf40144ae8907641518b75b94b25c73a8c7e73b50f21b899 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-7c5d5fba | sha256-12145e47c57543f37a55fe2983e5db63860942d66b188aee419655192b1636db |
