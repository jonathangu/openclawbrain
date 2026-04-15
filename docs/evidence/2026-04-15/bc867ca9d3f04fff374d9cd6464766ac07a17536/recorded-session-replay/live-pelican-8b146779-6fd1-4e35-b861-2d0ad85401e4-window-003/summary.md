# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9ea17f3aebef4af75f0a93c521d6e776070c7076063ed113dca780cff0b9684`
- fixture hash: `sha256-e43f09daa5c7f1f8012274d4f09baa27758aaa51c3e914baa4ee6b5329b895af`
- score hash: `sha256-b3fee9c2025b6154497c9ee9906e9f5a31f2df58ffa43b5b8d774f20dfbb3316`
- bundle hash: `sha256-c460d4789a51ad8b802f6ffa8201a6f69fc9578f817e29a035a6ccbbea52a868`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-174fe02cb9d576a687ddb560851b02ab0e12cb6737fa301408229ad552fa41d4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f590f9fa747b23d10dbcb24f763079714aea7af072f0cfe8084fad3e77992e61 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a1eda8d209a0d5f654fbbc3c5215234d0b732e62ca700b949dd8a89b8e3c948e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-23704188e738ed76da0967dd0bd23d9fcd315290976a7f9e3e0922a4c6e3a3e1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4563a9f5 | sha256-e53fa840a92a1efe01bcc666c660d07db61af2fc50d94dd3f337f54c223f0529 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4563a9f5 | sha256-9faf82243254a961cc812c3c0e5946b5af6bc5d237e2876f9bf280949a8b98ba |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-4563a9f5 | sha256-e53fa840a92a1efe01bcc666c660d07db61af2fc50d94dd3f337f54c223f0529 |
