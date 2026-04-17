# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15f74481afa0ad3c49942a752d93fa21610759dcd0f5184c05ee667b747607b5`
- fixture hash: `sha256-27569dfe07b6cf66e357fc072347afe0c073b0dd225ff6f7f6dbd4f6b53bd5c5`
- score hash: `sha256-9ab89733c23230cb82ad25eab1c941e8c31631a8e389988946c4fe5f3af9d298`
- bundle hash: `sha256-f68e3316f2d1ce42546046698d7d0aad6555090ed290bc9b0d7933502cbfd361`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d9780bcb02b4dddac9cfba41582ad72477a9d4e9b030a1ad3ced919c347c5d08 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-304ff52e65afb9245b5306081de69c48114cd8aa87fcd9aa63962d312bbfe6bf |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3905993aebfa7219e53b4f6e241687c123ae44994d559380247fbc60e5e99bf8 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-139e4226c3236c09eb71d36e87492fb89d0dc54dce0391f7a592fefd65ada9a0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d527c513 | sha256-19b695e1a8046335fda86078e3b9b874721598cd7bfe4ce872559dcc4a306613 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d527c513 | sha256-152d86a2570cb7738c2e7d5720c83869ea88bfc4a51e9817e247964beb255f48 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-e8ce4172 | sha256-414185f2cc1ab9f4562d8fbaba8ec9effba979e0e64ba6f45baad474d68b8656 |
