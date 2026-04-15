# Recorded Session Replay Proof Bundle

- trace id: `trace-retrieval-proof-hashes`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0459c0fc86792c696bf9cc08f781d4cda302256c62e036040b83468b7f5434aa`
- fixture hash: `sha256-037ab45cee260e72054bca957f17ffcf09f42052b826c8be6335ab9bf5a8cb0a`
- score hash: `sha256-4e9fe403b4a638d72e2cafaa2b6e4a0edebed70c891271782da1112d487758d1`
- bundle hash: `sha256-4f72fc56b2388f1f90654b877e7c72df1326203f388aad0bc79e06205608e224`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 15/20
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/5 | 0 | 0 | 2 | 1 | 0 | sha256-92b6e3cbb25607cbcf4bbb960bff7a027b61da7359fed6908c38ec8cdeaca5c0 |
| vector_only | 2 | 2 | 5/5 | 0 | 0 | 2 | 1 | 0 | sha256-927c15bb482d2c40af3d8cccf033e3727520e9ff20380f730530b90dfd640ef2 |
| graph_prior_only | 2 | 2 | 5/5 | 0 | 0 | 2 | 1 | 0 | sha256-ac3b68325157f0d774356f684160513b505bc728a55a1c7a201a4bee1df21611 |
| learned_route | 2 | 2 | 5/5 | 1 | 1 | 2 | 1 | 0 | sha256-dca9dbd2e97457b02f265deef18d516dc975120856aff0f1093f9b26a902a336 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | proof-hashes-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | proof-hashes-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | proof-hashes-turn-1 | 100 | yes | 2/2 | no | no | pack-3ffcdad1 | sha256-cd93181425644fc27b767a0f99d0d1a961550f12bd4a81171cd113401013857a |
| vector_only | proof-hashes-turn-2 | 100 | yes | 3/3 | no | no | pack-3ffcdad1 | sha256-cd93181425644fc27b767a0f99d0d1a961550f12bd4a81171cd113401013857a |
| graph_prior_only | proof-hashes-turn-1 | 100 | yes | 2/2 | no | no | pack-3ffcdad1 | sha256-cd93181425644fc27b767a0f99d0d1a961550f12bd4a81171cd113401013857a |
| graph_prior_only | proof-hashes-turn-2 | 100 | yes | 3/3 | no | no | pack-3ffcdad1 | sha256-cd93181425644fc27b767a0f99d0d1a961550f12bd4a81171cd113401013857a |
| learned_route | proof-hashes-turn-1 | 100 | yes | 2/2 | no | yes | pack-3ffcdad1 | sha256-cd93181425644fc27b767a0f99d0d1a961550f12bd4a81171cd113401013857a |
| learned_route | proof-hashes-turn-2 | 100 | yes | 3/3 | yes | no | pack-812c3858 | sha256-a7ea1bd8e812684af4c0e420308345e83b5ae469ffb1b77d9c6f791a9b5ef892 |
