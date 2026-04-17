# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-821174c9e8199055ec82b211fb2d09a993a3651df37b6c8cbf4707f78e0465ca`
- fixture hash: `sha256-bcd47938b5aaf944e8ae47149b98261af7f5e6b78cefab4ab8c21ef4d0f8288c`
- score hash: `sha256-77a1c1c3648bc0237e385c6d4a82d1a057131d9465c3055d991706e1dc153dba`
- bundle hash: `sha256-258d0daa3b25db30d8eebc126429906c3f3d0b684168bd1556964425d4c496e0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46bef670c7b3d0d500dbccdbaeb44127bcccbda5425d78ea64b9256410c95a9e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aabcd4327e8fbf85291276f7b83a64b22236ae21668b2bdcfac9e1f65a3fe5fc |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a9cceeb5aa4277a32d873a0076d9291c3cfd83c869244b78b46058e999fec787 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-87c3587364f9607190e3cc554b0521bcb5e38f396b7094df6575664628ed0e79 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1b2aa3a7 | sha256-d2f1f4df587d996e5dbca54cd62c278797e231a43cbe967282733486ce6eaf19 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1b2aa3a7 | sha256-903260576f56b2551fbdd328dfdf266aca17ef5faebe9376854935c737755898 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-32ec3c14 | sha256-6fd3e161221e3df9a4f60e0d9df9cf41d2464ff0c1f5d30482be35b034afa555 |
