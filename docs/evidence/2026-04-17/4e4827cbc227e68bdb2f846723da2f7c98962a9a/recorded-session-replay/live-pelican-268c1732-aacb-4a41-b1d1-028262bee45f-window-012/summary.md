# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d0b5d294f5bcac07c81e1e9b7fbd08fc02ac60b4a0afd2bd2ad3564216748c02`
- fixture hash: `sha256-10c2797f7098132dfd19e74471fe861e4fd990acaf92ba667dc395a281a0c32a`
- score hash: `sha256-a989d4133495492e1891271c3358f2b7d85df6ee00c4c2736606e85274901fc1`
- bundle hash: `sha256-ecaab8534f90e22fd1d4664ab45787663b2d1b99455944abf2905d56a1317ef0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eb29fb5cb8fb01ea6e12d04715c0ac66ad31c35de2501ab2ab9a23569a1d387a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5b8014fb4e258ba725bfff14015f6835ff3972dd473a497e48a4ac5e2f52f956 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6ee8094cd732ae6688dbef56134932bb0f16050f1372934ac70639e2554540d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-9f1a0ebe442288e9e7e9f5eeb36ebd3b0ec80d802c382ffb9f6071d4d23c9ee0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dcdbc1c5 | sha256-27aede3bc60233087b29eceef0c83f64ec8ca09ca45535f933b90bcec5e5234e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dcdbc1c5 | sha256-ede211c35eec01aec90b60d206fa2394f82927663d9119c24a433f0d48f8f9d5 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3218bbae | sha256-fd13538100e3aa02e4479f943c555d2cee053d0adb96e99b954306b25d460da3 |
