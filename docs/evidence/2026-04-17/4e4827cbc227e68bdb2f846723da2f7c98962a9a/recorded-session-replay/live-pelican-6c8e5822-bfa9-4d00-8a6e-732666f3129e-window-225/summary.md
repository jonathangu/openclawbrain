# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7768c82b82cfc9e79b11d2862950b229d848ebccb180ebcb1860140cf56b1f18`
- fixture hash: `sha256-e3d4346656fea9fcd52a8093d89ccf43c79e719fec02594aace8851b57c7f190`
- score hash: `sha256-726dc417e5855560d39ee703db1c9c2a0d1d89ac0d5590a9325ef3437e450abc`
- bundle hash: `sha256-589927bfb6c2d31ad5fa3dea1d5bdd1b9657fba263a4ae443179193cedb22428`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb4473107ca7b170cb7198e9a132dfc26d383b8a4567d404be160b76d2d08390 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4f118296f40df951cc1dbd4c41784b6ef5d1cd289dbe22077bd1958698377196 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eb4def056b316ae7e9a62086e0a93880f8106f8d1e38933d434cc9b10db21c21 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-44a0c00fe4fdfc6ed2ef51a2ebb161a46ae03e510c7f8bebc5e84650da82aa45 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2e019157 | sha256-37819ca6bcccb42df068b9957884c6c6ce32befecd91393706abe1fad788787c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2e019157 | sha256-ad0db728756321873500223c4c74d471f23ccccef1eb44d48bdc75dd21c59578 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a7fe564a | sha256-f79e88ebe2e2098eaa57b0fd0e27b412e36c4d34c9e97f8fb11eeafc5c880334 |
