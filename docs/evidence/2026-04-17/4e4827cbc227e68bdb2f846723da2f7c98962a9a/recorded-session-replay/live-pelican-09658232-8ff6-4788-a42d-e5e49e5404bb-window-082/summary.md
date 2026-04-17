# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bda6b3da4ef39b29be45310328eb0566a39d316769663e62675a6105dd7880f7`
- fixture hash: `sha256-e12b530a582d1487040cb7cdaf3e1255576e9298c334dbf79363d1f81080b1c8`
- score hash: `sha256-c57ca79f0a5ef6bfe27204ad47051f4589fd02db6a55c3d1ec308bab75c5cc3e`
- bundle hash: `sha256-8836a5dd3c71e0bfda4136ed013739680c3deac58f646c1d75cf8e48f649c24b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bab067ff9232cc412579013f9b35dc498686eb53e7f83b8de58e12e80ba3c742 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7571bba654406d093f901df152726be5f20238ad64ca28168e65795214b5903d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2aa20516057688ffc95bba13d6b6744a2200c6b6c327c808de8c31466b937387 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-80d9ebf40ddacd8febaf59aa48d14270926db9ce786d668e6f6eee08755d9c73 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b7588772 | sha256-4dccf452423cf25fd508f8ae2f48064d0047a8deed0e09cba144fe9655552bda |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b7588772 | sha256-2870611824f1b33896016f7aa1d0eba595b97253a44f2ccfb430f3bed2d20eff |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-a56adc91 | sha256-3fc734845384f703ae4925ed57796ed525e9dddc359538fa9d4ddf368f25728f |
