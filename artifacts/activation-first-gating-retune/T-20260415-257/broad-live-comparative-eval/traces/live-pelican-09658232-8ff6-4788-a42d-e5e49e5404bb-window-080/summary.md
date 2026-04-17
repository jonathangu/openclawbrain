# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffb842d2d3e2bdef797f256817f8e1d78ce9bfb6aec6432fc2346aa3c074ed92`
- fixture hash: `sha256-198b6d169e431ec0de7f8f7799921b3db142fd3140222b6f3b7adb7cb8af186e`
- score hash: `sha256-9e9a8d5cb744767fac1213b72dbf4e4dc843f00dc87eedef84cf745be4d5bcc8`
- bundle hash: `sha256-892e82c7e268ceeada96a1878c7a1d3408a0dfdb053851d2ef5702f45c4ff4a2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-236480801c2c951ce502606c7421d96d831d23f14e852b882e65d08e48147fac |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d150a575d4b34c894be8004b28ddc85fddfff241c930caa7fac8ebd2c5a5c104 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-98307b78fd7389f8f48b03e0f817bcebab2e9b3b8ddcf19931e01d7b09b19e25 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f768ea9c937ebc3d82f049e1994be4c8e8b5a4bd471fd2d48061568e49db9ffa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7b3b9835 | sha256-5b0779e8d08408a22d350f3f48b0463f4d325f85b319815300cb6c0d9046b257 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7b3b9835 | sha256-c72bab983881c04f93c5fb69b8b496c7a1ae4ac590fa7d944ed2b94b6b58fc6d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-a8c67cc2 | sha256-6b5729f116590070e2bc0b7632b778be91c7e092816bd37ec3a0e8f7cf9fe957 |
