# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adc21e40f3c3bdc2111e458183ef292b9fdba4cc9072a5e4575150e3a25e7599`
- fixture hash: `sha256-82594518eb539bcd92075469119fdd7049793972cdce0d3d047ffdabe9e539b7`
- score hash: `sha256-8da460e3c820c2ce8ce3cb628c5f72b5f420ff1f0bace3e8365934798d6c549a`
- bundle hash: `sha256-1414b1b87e98a004e852054f5664ac864f71eeaf43d1c4f232fe3c1f840514e4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca48aec6e03fc6ebf10d02ee2af1729bb6ff692653b0f22ac3e3b10f844865d0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ba6c1f4d176c1c7f79afc865d563ab6ce1074a07d495b0b5379703d86a16a08f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a9285c860932785b143a0e7c1d5e454abede4de3a2c33c52e9c1284631b9b93 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8b96061918af593714e5229943659f23e8b2aa27967197d9735a065262dca083 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3358d834 | sha256-4f854882a13fe8f4b27a436ea02f284cec112770b8f9bf792d142d97b880f872 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3358d834 | sha256-9ed8d7261e2bfe903cfab61690d06f5a65e7dff5bcda99d70aa0a585301d22de |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-5e412e17 | sha256-1391cf55c7c963c560e82c73493a2b7412d0d794cf2015c628f934285a6654a3 |
