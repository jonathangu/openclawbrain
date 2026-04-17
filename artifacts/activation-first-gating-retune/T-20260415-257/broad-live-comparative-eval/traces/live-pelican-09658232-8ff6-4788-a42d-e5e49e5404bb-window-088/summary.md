# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e0cde3a25a3b093a3328111ec29f970fc82068378dbcdb19446f77be2e4c1e`
- fixture hash: `sha256-a657cfcc3c13a64972df27ba9b34b582252db2226ee691420ef45e3b6a2bad38`
- score hash: `sha256-b42b92144a2e00ab87dd965ba64ccb949f88998d61bfda13a0d2aea63652fa72`
- bundle hash: `sha256-99ced4b2fbd57ee9c8b95ba7fdd4ff389ee3b28679542b0303556e853f42abfc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0b3855bdbfdfdd31f2cf7aedce5b8a8e42a2e757ba398c67ba975aa86dd21ea |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aa68fd11d29deffd175ae7d260867413d7e2d98b3a2a82bac39ba68fd0efb118 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d7c17ae950086d0aaa3dd6ebb06e6271d9cb7d80d176a448811e86aed0d882a3 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ae6eddf1a0f94cd97213a6d6f2c7ae1390ad872d445e611acb5b376e1651b238 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc3da235 | sha256-4a56905bac5c1ca0c0d8e9a452f12f86bf0ce6abea86ba32ef2b4eb1de3757a3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc3da235 | sha256-ee3631f3dad1befa7fa0e4c63576d9b01299e660b25873e0ece24acd4f4b7c78 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-392e9f98 | sha256-68f1eec690eca61f613ab58fb62b33913d7373cde72945f0ef0e32385e8a5c8b |
