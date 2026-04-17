# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ac1c0fab25c05525176cdfff2149e8d15cf9da0d9dd3e3ff8d1e6b40aadd074`
- fixture hash: `sha256-99a9dda4d1e27d20e5b5802fe99ae2cd9ee98cd875422b1ef45282c42f60a797`
- score hash: `sha256-155094ae52e5b86923f6920666f7eccdf7329d5d4db3096a7a2278f3eb46fed9`
- bundle hash: `sha256-d3a96104c1336748b4c4eb159326a4e738f2aadf49d534db7ebca15560398d29`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c3402e859f2552a40a7f253ef60215bf90d6f117858139b3ed26992a03a4545a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d69f48ee9d476f3f46ecab415d28ac09bafea06adca0df2a3a8c91ce95d44ad7 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a623997a552b5d07db0cd4fef0f6b74d770e04404680b7ebf867b11732fc968e |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-f9197b448d920d35abb8dfe8b4fc142188641f4d3d1553180c3833d017919218 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-965a2b52 | sha256-3bbc8b1b44b31768203c4738261f604502690281d329154825772a303c390311 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-965a2b52 | sha256-7b70693ebdf970000d4e5aac076ee741a9a78f1f08942eb7de100c3ab6b2d843 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-e69d681b | sha256-db76a7a1edb2c1e7a1f29b0cb0b1b6f0cfb8c773041ebd082070b012474eeadb |
