# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37ca78c4f79af1f5ccebd457f09d9bd9f0b270ac0d1c7dc3ef10aba20d199a04`
- fixture hash: `sha256-221b36f5e3c3b83dde39237b8133ec3e68acdd74bce0b4e3672a3fac84a8cce9`
- score hash: `sha256-690595dcf6d60911238449d2dadf8468da3082ca2949611dbc4ff39401c72b4c`
- bundle hash: `sha256-807cc6b29882b20e7c6ef3798e91d7ce1a63657a6d5727f0a206f90e8f7b9c16`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fa6a30b0b756b7163e1ab0f1526218df1fd81b134bd908830d7627bb5155f717 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-745355cdbe6f3d16678ca6b800feb20ba5e8bab724589229bbb314b5227c3912 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-668c2d4325f24f13cdfe6ac892a8b18101178c2f97200606042a56119fcf0f1a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-cabbdc44e3b44d174f748ba40df04da5f21e7bb9c99618feea96f620187c9f0e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-07a88d40 | sha256-70b6114d86aeb4d180545063519d069e980cf5b6fd5af6c17e6bbe75b6f58a6b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-07a88d40 | sha256-515a1d0c254a73d6bd176c1fefc826174a210b3dd9b9ff046e3a7ac208d23f9f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-fbe52583 | sha256-ae791c7162744d12ca0aa7b10b20a3559086e3a699d6a014577f8d591f2c25e3 |
