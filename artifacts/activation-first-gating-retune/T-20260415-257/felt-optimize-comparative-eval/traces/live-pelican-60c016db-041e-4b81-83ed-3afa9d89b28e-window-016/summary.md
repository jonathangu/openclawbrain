# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3356e25999580c1815d0ae49bf40bb5a370485dd768a3aa1572de8bdcc8cba97`
- fixture hash: `sha256-0729bfe4197c5261cfcc1c8ec0f8202300a63367aac183bee3a483ca417a77a1`
- score hash: `sha256-3b21fa36b7f5b54aa87e5244132bc7e0a6973288299486a4887d6e5647b596f6`
- bundle hash: `sha256-ed852719f9446a638134e6ecae65acc725b5af792c64e1763454829f36418e48`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a4b42315dd1b1a176628a7fe8f13cc2dabd4068509144906d5e0b01e2bdfcba3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-149158d823712c7ac5c997d55fa7856a58d0e5b559388bc8963fcf21fcf4f8e6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0fdf290056b254b512694dfabe1360dbf71ea619f2a99ebccb7b7edfa373e15a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-99a017866c82b1b19f843c80ac8e5d198eda16c250db4be61b2cf9d4cd5c3fbd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b833f4b1 | sha256-b75800fcad0493449a4a9cf2f339c8d9e394a6c89eff4dd58dc9eae7bc4e111d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b833f4b1 | sha256-e0efd38c54273510fb6ac0c0bebc1a29d3f4ea1bba1ff2246b802d5ba840a9d3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-444bb944 | sha256-a6bab74f294a4e7593c93b735ac66ddedbd0644ff295dd3accfc2a4f5b839232 |
