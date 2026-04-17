# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-76273bf572d2d6df7f5708306c7a325e0d7fc022256a8c48664d2d2c99f93d6b`
- fixture hash: `sha256-5c64fa2fb4319875db5a6403e087c56d2c1a468e1ff9a819a4e71ac1b0668ff8`
- score hash: `sha256-06aedd1224b3561f0388b5954ea2a714041731fab9018caeb91efac11926c1e3`
- bundle hash: `sha256-3c05d163890e43c07f7e2242911005577aad2aeec5a4ad71354ecc8cb8ffb3fe`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd39b973453e541375d3824b8f2b46f3993347e9ee385b937ee54648b5838113 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-84e59aac880ed4c407174aab9a0e72ae136f0de3ae96345a3476ad294deb116e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eeb779a34af9cecf3649e029f62f0ce1434218bf3b3ad29569e06145b0afb5a7 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c63db465a4cc19aed46756b4a0a17957d2079c5cfbb75ffbce908fb722b9f6a6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6b639443 | sha256-f51801c20f7881da6cfcc2b30100584874ac76b9ee747464a5ff57da6d138261 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6b639443 | sha256-f51801c20f7881da6cfcc2b30100584874ac76b9ee747464a5ff57da6d138261 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-018f2a80 | sha256-2acd83d7c4b63dc33182f019e92c2b8d0c0e7ec060464cbca6243906261efb5d |
