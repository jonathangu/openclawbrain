# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c58ed04d44aeb04071688c4a26c4c689e25ea007697f349c3e4c8fcbe3bda533`
- fixture hash: `sha256-ad70501e856aff4a57d924d7225c4dc64463e70da2f3e42777305ef85fb46a26`
- score hash: `sha256-283ed521162fe7ea149498ea96c024886d415466ea250a6edf586b541ab898f6`
- bundle hash: `sha256-3738a5dbe2e60e422ce1a4139a0d53e5c38c251aff9b50e4c93963ff570b14be`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d770aca06ab90e2e0a0ead714079ce642ffbbb18580e6acfdf4fde922a74f5a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d3f1bdab1ee9787e1eb0e47d4decf23caca748ee038d06e24185434181cc9ec5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-29d1eb80c097baddb88c8f7f47c99b806468ab6cfde4a894cb2b9a570b57c1a9 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-64b50203ed3c7bd3f437b83e77f05e3da035e87499e7f02f0aa6ec408fd6bb1b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-77ead2c8 | sha256-362ab3a723317eed775aedf891b1e3a1c389cee075fb3c339fdc166b3b5d48ef |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-77ead2c8 | sha256-7263e5acecb75856fd036c63bc55f650095a0dd3b570ae4cca6811ea1d69050b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-77ead2c8 | sha256-362ab3a723317eed775aedf891b1e3a1c389cee075fb3c339fdc166b3b5d48ef |
