# Recorded Session Replay Proof Bundle

- trace id: `live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d851cdfc065d530ff6a05cd12aae1453cc6c5cc252f286f05c63b39f7b7ea103`
- fixture hash: `sha256-add4e01555ea0b700f89e1179ee076e863d3216d180ce57f607f066d853c468e`
- score hash: `sha256-9d3eb2e125c17a138c248ed2f076bded5998be559a1f728f9bb8245dc8839d4c`
- bundle hash: `sha256-6c1a0a97d60362be18bea9750871cfa67dda86ceb8635202a950daa555cf2820`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e16f2c51fd8866c40ce249b661c20fa44d3a586d3c45a550284b22e35e90bd83 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-955fa5004dc6fe853eaa3bca19c71259f04c6c37b05c07b8e2837f819ddade14 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-af97f1c8a857898306f76007a4454bda79ae63c045884ddf5de0ce7706db1270 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4a9f322b20209a294c5e7cb99578a10df8a95d81f82954ee1cf0ef19cd7fe965 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-aabd5a0b | sha256-35841642381bd153bb4d1316d4907371b16f29bff3ac40a218a7f5c848d32279 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-aabd5a0b | sha256-ab8ba3907442b3fbdeeb0e7570cd117a1484f7f7b7f6146206d1b36a9a563dca |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-aabd5a0b | sha256-da5912e0dee4ac66f6db6d5cd2942b43e93dfaee7415b5f6c9906802661a3ddf |
