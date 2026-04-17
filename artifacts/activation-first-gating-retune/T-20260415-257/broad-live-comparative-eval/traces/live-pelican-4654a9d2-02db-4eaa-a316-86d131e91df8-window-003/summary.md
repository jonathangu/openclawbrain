# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003`
- winner mode: `vector_only`
- trace hash: `sha256-871a8d0a2f1d4e43acf8de9d8e6956ae4d1ca9dd0a419c5265c96970bba52722`
- fixture hash: `sha256-219199343b7c6d3ad1312b7304ed4e0c3741109cf5c94240ae657c56e05e2f48`
- score hash: `sha256-0ca16841ae77965038d6d59bb0ae7a86ab79b11517721b1281df05b291374459`
- bundle hash: `sha256-9b05b36e5de4270ba15210f582cb5e7df6643aaec2c67ab3565f50b8899992c1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 70 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/8
- phrase hit rate: 0.125

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b917e276e14b121cc269645e14a5fdafe3dcdf3d48a758ee09ff6c7e3bf5cdd4 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-066d1affdd0030a848f59f453c656ccfcf025bb6252afca5401f48d787840a94 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c103b8d08c639b872d2dcfc5fef1e6dd20ae58b22170a5a0f1e5d03570dcccda |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-2a2d13e2b83ac0b263e7bf3fad98ea635d00f748623e34c830498a4a9b8643dc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-ae10a1b5 | sha256-75710054b6c046815b2f15c0df87902b3ed7fa63f4029627f040170cbef02e2a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ae10a1b5 | sha256-bcc3f539465d51383d80a42dcc4924918c8905ad3d692eb9883b3d0ab314cac8 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-a2dd9b2a | sha256-2614bbfa38c12e654786e1d93e82fad61a0f8dd505c304d589109c6313d68ea1 |
