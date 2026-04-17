# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b29852ddcae763768818b925bfa513423dbca5c8ad934450c78f0838b90cfab`
- fixture hash: `sha256-f828a5ef63881667b78ea5f5530e5417bb5590176f57bdcf8c4590150136788a`
- score hash: `sha256-1aca0906fe44291d7327d6e15bfee54cf859d3d0fa243c09290c7f88dbb5d776`
- bundle hash: `sha256-f9b023bdcdcca3601afb94dc175cf3f1ce3fa7799f9dfd2a2284553de8e7094a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d23197d4b519cff22649347398dfab9ce049fcf294afab672a8b41fd8ebcbbad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-26fef0499879ceaee1f35e09beae2131d50f0839279b6e322699f5dc3d41192f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a26eeb825d8be0860060e2c50f2eeec41d4b7c77e1e1ff553dca980bb795872c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-a30badc07af89b07f10c0aed194e1083743f7a815a25ff4fa3c2ef512a232c3a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e5b91b90 | sha256-9af4f71b41afbb387be978c98b435d9d4b05ddaae885c1e61f2c0ac3c9b85791 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e5b91b90 | sha256-fc83d676fe89378d28217c36c5149d13706db53ddf97eb42d7a4888e8c22d768 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-a5bc7dd1 | sha256-b3419a6d5e4d7874beba02fe7e008fcf33569e4874688a1cd859457adc5f2fb0 |
