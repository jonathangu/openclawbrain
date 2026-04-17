# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b973c64c8eac5a0b6deba25fbae9f31be4599e3d19192c7c9dd0b18e718f1e`
- fixture hash: `sha256-b932d5e627b7081f980ab111b252e205aa7e0185bfcd774e6388fb9e948098c1`
- score hash: `sha256-6caeb0864c4a7042f8d9f3f24c2f426a75e8f0bde5b11c0051669da3b4613f42`
- bundle hash: `sha256-af42220501d70218d398f285197db342b72f4ea5f1970b53b1c782ca03a0c374`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fd3c28bfccf2817f3d01d14dc16c97875abfde806e8cfbeff2d04b6e2a397e7b |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a67df56a88d673bd634565c897ecbda22ee2ac8bbcda6bf00fb81bd910c6ebe3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e2afbf4bb6268f90f09eb5dc80d394156d1b4fb262694f7696359ce8db955aa5 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-bf64115cf9ce5e0a24fbab24184b7942efc85aa49a8970c54dbc56d30e947e7b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d40a409b | sha256-5b8ae5489645f006edbbb3b8716e488f63859ed03968ac7944978ec755708189 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d40a409b | sha256-5b8ae5489645f006edbbb3b8716e488f63859ed03968ac7944978ec755708189 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-a9ff248e | sha256-60d76823d169e113924991ff039d8b702fc41813909e7955fc65d66bd3cfcc62 |
