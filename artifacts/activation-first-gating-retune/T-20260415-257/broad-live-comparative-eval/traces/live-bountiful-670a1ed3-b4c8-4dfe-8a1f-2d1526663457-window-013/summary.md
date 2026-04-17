# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ad8a200767aafe991ee7054e677a19f37758804d5a9a487f59ccad4263c83187`
- fixture hash: `sha256-23ce3445f512fae9ac35202b97a34c12c8d0db3c79197541a8b90358597638a3`
- score hash: `sha256-eac75c6cffde6d94830454ad7f6a5ba183fe97ac00e4427a26d487b618cb5ddc`
- bundle hash: `sha256-663e5c633d3decb176a847ad55f40a9b746b30e77bee3777797e248824afa1d1`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3b757ba8fadc84f09e0e7aed31f0b4ebd54fa8fe354fc559aafe046aa0541083 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-59ad372ebc7bb992ee0c8155ffbaa8342a9815fd212200d21f957d6aa50634b9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-674ffaa4b2e79759f021ee9ce3b19c27d7ce662b995c462f0a24c6db66c46e5c |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-6d3555fd56d8468a86be2031833693e2e1b82a7b86af02db1311244dbd18ab7d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d476be2b | sha256-2b73608e8c3089245448f279c557a67c7dfd0fafc51ad529ca462ebe6ae8ad5e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d476be2b | sha256-890dde78e4637a618c2c9ada93d977a8a60d9d02ce05594c761548114809b848 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-5f3bb4c8 | sha256-ba7233fd99bc76e973b3003139f9d0500b55222b0d534b02806a7a63bfd1a3a2 |
