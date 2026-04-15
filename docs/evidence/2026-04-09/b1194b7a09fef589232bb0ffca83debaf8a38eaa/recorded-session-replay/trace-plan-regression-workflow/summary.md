# Recorded Session Replay Proof Bundle

- trace id: `trace-plan-regression-workflow`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d6c19595e20ace6fabb5b7541429491ac83c38d9e852f37f0b5ee0bf60e6ed38`
- fixture hash: `sha256-398d90b9686a4dc952239f413c84c29595706a3f11e555898846a0d70de73153`
- score hash: `sha256-3c6124f3ea39c2727d6a430e8fd5a2b3ba3cf196467f9eef155dcd303cc462e9`
- bundle hash: `sha256-a1b5366779dfcaf3ef68b7d540653366bbb48126db816059c860e587ee706241`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 88 |
| 2 | learned_route | 88 |
| 3 | vector_only | 88 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 12/20
- phrase hit rate: 0.6

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 0.8 | 0 | 1 |
| graph_prior_only | 2 | 1 | 0.8 | 0 | 1 |
| learned_route | 2 | 1 | 0.8 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/5 | 0 | 0 | 2 | 1 | 0 | sha256-90328e9b0c4f9b38528d7a84c935133ee1cb08f73bd2c39b66d2b7ab6828c750 |
| vector_only | 2 | 2 | 4/5 | 0 | 0 | 2 | 1 | 0 | sha256-b300a8da73116d456349497cdded2ebe7ce0bd982c492f65262515a40ccd66d4 |
| graph_prior_only | 2 | 2 | 4/5 | 0 | 0 | 2 | 1 | 0 | sha256-3c63ae76c6b2de702fe89dd48979f71e7c120dbec6bba5f8e1a511ab91eb35a9 |
| learned_route | 2 | 2 | 4/5 | 1 | 1 | 2 | 1 | 0 | sha256-7379aeba5fb7723180f55c03b12a8f3bce44ed02c72c41a1e7d5d81dd735bcd0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | regression-workflow-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | regression-workflow-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | regression-workflow-turn-1 | 100 | yes | 2/2 | no | no | pack-f6adb55e | sha256-404470f678139cb4f2752e33a8652f41498f1374ea2dc09654ee3feaacff66dd |
| vector_only | regression-workflow-turn-2 | 80 | yes | 2/3 | no | no | pack-f6adb55e | sha256-43f2eb6f5740cd0cd94c5af0cb550e24158f949e6ea1bfc4ceff7aa445ddd991 |
| graph_prior_only | regression-workflow-turn-1 | 100 | yes | 2/2 | no | no | pack-f6adb55e | sha256-404470f678139cb4f2752e33a8652f41498f1374ea2dc09654ee3feaacff66dd |
| graph_prior_only | regression-workflow-turn-2 | 80 | yes | 2/3 | no | no | pack-f6adb55e | sha256-43f2eb6f5740cd0cd94c5af0cb550e24158f949e6ea1bfc4ceff7aa445ddd991 |
| learned_route | regression-workflow-turn-1 | 100 | yes | 2/2 | no | yes | pack-f6adb55e | sha256-404470f678139cb4f2752e33a8652f41498f1374ea2dc09654ee3feaacff66dd |
| learned_route | regression-workflow-turn-2 | 80 | yes | 2/3 | yes | no | pack-a5e00ddf | sha256-23cc65271d24f6cf6c2ac8ee93e7ac82a94b2c64e2000acf8c34ca509a93829b |
