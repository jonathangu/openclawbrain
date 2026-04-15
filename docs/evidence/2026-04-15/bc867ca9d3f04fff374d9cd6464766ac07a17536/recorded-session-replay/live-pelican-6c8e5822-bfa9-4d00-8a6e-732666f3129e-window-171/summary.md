# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f2aad77541ac9575f5e5ca17b331150d26a5ffdab9f43024542cda1cc603e5be`
- fixture hash: `sha256-bd1f8b0e0683d35bf0b6cddabbcb17bfbeff749dd6d56a3da4fa75988fc68560`
- score hash: `sha256-ab49ecc694e6458a109f3abc6e91ebe9f94de2ba9edce4cf68a41ec0482303fe`
- bundle hash: `sha256-e30be677fece4f2a338b30ba6523a1808cf8e743137b6bff7ed574aa8d69f839`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b6dcd51a56bbf9edfb3ea54756a6521b5761e2fe2a8b04b095719a90cd986e9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d9a490b75bcb23df42eb8911b2b2f4826e0f2c93e722f5eb8b174c4b767f530 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68729c2cdef20aa7590dad11046f1f2b66348a7b4d4d705d3c29a2da9c9dff8f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c2f9b22e4e1efc627b2623030ae883a9919b5373dad81390ca048c287ba65b4f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec5ceabb | sha256-61d6c0f31c0493ce65f218058854294d479c9de95955059d67dadcede8adef2f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec5ceabb | sha256-7d9e0c465d60aaf60c4a4851abb0aea3d2e565817ad71c09eb79f191962212ae |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ec5ceabb | sha256-61d6c0f31c0493ce65f218058854294d479c9de95955059d67dadcede8adef2f |
