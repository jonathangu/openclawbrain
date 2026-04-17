# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-580d4f8ccc5672e0994a6e33aa91736865fe4849bdf6f4307be6ada1929aaaa3`
- fixture hash: `sha256-f2131456430264646f8b93eefc85baa48a48ed730efbc3d47ac8e04c07a9e06b`
- score hash: `sha256-e646d7e8876b83dc852a47aa2130f0862e89b8281c5f5649762642564334f1f3`
- bundle hash: `sha256-b752be9d0ae3027c37a7b8f1dc13a6701b5753d27ca40adc873bbd336bdba9dd`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | vector_only | 70 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f4cf4becbb2f48d6942aa18f5479385f6a9e60c52e2114c0c244447f1e9cb2ba |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-9c09c360cc45819a25b7908a13f28e3c0c0c89399bd5cdef79a8d96644fc3f84 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-50f2430f499cfde2e7e0492156ef73571d42c1aca3aca8b230b46ab69fbab522 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-5c08b5da729617c1dfd56a97740ac7e0c3565f9482e88829289f0879f9768bc9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-3d75b303 | sha256-d3905fd01e5689b039d850e2d38ccb40855ce8521af4879b622d8f163204a511 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-3d75b303 | sha256-154ded79194f941fd5f3782fb6776035c3c784bf95dc620c6749f438f5bf4d14 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-ca601306 | sha256-2346f182dae04e9f611c8a0c24a6b956b2861de2a860c0f20157370bf45140a4 |
