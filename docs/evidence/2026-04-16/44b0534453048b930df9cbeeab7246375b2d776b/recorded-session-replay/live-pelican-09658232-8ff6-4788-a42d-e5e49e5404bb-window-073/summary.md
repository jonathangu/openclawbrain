# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4344938604e067860a8ae5cde1fca1ccd4f50c2742543e1ed5dbbab203e23d74`
- fixture hash: `sha256-835a3394dac9a8b8023e71ca801b0ad86f7853cc9e826a2ddbfdbf3c56dd351e`
- score hash: `sha256-327bab9ec5c4b33f619db51369ee70dde830e4c28a354eec18fcbd84c50cb3b4`
- bundle hash: `sha256-cd9002cafffdda7c468bcdf4038580d1d8bb6c63d7e34173abcaa938566d6d26`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-95f6b633b00bd779574dbd24baa772f0fb4eebc8350ac2c13ddc54230525a7fa |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-158689181c2ede4c0a7cd3ba99780211f87129dd900317073588ffa9c836202e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bb1c0df85301b92fe9b8f4d64dc59404530c7dca1de4d68a5364b2a7b5aca6ba |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-b5b23cb200e073a2b8b2fd218e0338b6cfca2a8d6bab0694cd104147773526dd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-de89321d | sha256-a8adbfd8d9e1663f1353737870a2ff9552a3978c7a61aa11c27ca385e1dcc9aa |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-de89321d | sha256-446148d5eb55f7fa6c96653885869acc2024f3cf9040e66b643438106c3c36d0 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-27652ddc | sha256-755dab02c94c8304a3fae322ea3487956e1a5bfde8107620def27892a5db4938 |
