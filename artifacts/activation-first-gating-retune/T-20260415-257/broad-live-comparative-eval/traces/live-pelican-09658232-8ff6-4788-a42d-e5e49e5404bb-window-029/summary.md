# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24701235d9bef68e6850974201809e3a73463fe7ddfd0b5cfe74a867885dc71e`
- fixture hash: `sha256-7c9db0ae094c3de40db6d4e0f20c52b15a3dee97c3144a7a4c433e3dd89b20b6`
- score hash: `sha256-8aef0fbe0ff5858e4fa048790c5fc44d3546f4e579717daf0426c4ffb0fde5ed`
- bundle hash: `sha256-17b64aadfe7746ea680230d19a6872afad79ba31b43e02f7e51a23260b9e2ffb`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f3c8d6c272d7556d73fb57fae65bea8046db993f5ac8290705eae6ece09a508 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6d9d16e8d38923bd8964cfb5fca9f09b4adff5baa994ef06bcd3460497d81003 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d3c8b364b5f22b3899e08d95e0236fa4183a7aa27feab8ac23103b1f28f95137 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-bd62e046a709f4fd8570e8f09807fe5bf1b3d79334ea544056840abbcbe38b7b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cfd389c4 | sha256-c30345e7c838eca9ded8827cf010a64de3fe8a9348f74606936d147acf410692 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cfd389c4 | sha256-06ff7d4428d5c0225ec6aafaa555eff3b2b421a4cb709e4505e2791cec35a47c |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-6d5a5977 | sha256-dd71b047d9552d0d0a41f6d02e7e40b2d3f57be11771ddd841b3a6ad047cdb6d |
