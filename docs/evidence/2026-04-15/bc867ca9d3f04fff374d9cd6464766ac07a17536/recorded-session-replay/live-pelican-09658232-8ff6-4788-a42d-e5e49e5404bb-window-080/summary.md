# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffb842d2d3e2bdef797f256817f8e1d78ce9bfb6aec6432fc2346aa3c074ed92`
- fixture hash: `sha256-198b6d169e431ec0de7f8f7799921b3db142fd3140222b6f3b7adb7cb8af186e`
- score hash: `sha256-f96d21c212c06c6b1ad69864ea7cb3d957c470a0628bb555f8d2acdb0d02da4d`
- bundle hash: `sha256-894fdab0d40518afbeef3607a8ec1179eb64ced310e849953eeff8714e99f99a`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a09343e41d242290465c88824ae6ec4d000aa19b0067121f88752d0b390596c4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e477801a555e6e73faf672060b6c0240e662e292aeaaa3e65348ceb9b1cb74f8 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ff0e83c3f8d8d7bae057c8220cd21d2aa79b3e3dcd3a89b6cc67eaff85c94c9d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8db20ddf | sha256-425960222536196155ef53eef7dc5de25c0688b4177c057d310b380b199ef5b7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8db20ddf | sha256-96bd6b0cc3e07f76a9f80c12b97dd80a0e1ab7c480b9fa992064d03b6b470c16 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8db20ddf | sha256-425960222536196155ef53eef7dc5de25c0688b4177c057d310b380b199ef5b7 |
