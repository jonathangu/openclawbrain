# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3abef879206da47064eddd47e25da3ed69b90db7cd3c4a8ad4966415b7f00bfc`
- fixture hash: `sha256-9918ac1f02e6942937a0c165ef4e1221b4c237d331f00ffb8e89f19fa2868433`
- score hash: `sha256-4b81f8044d41797faa07b86e7a519954d69317e58568a0185f836164edc6a358`
- bundle hash: `sha256-e90bf4d5c191055e0e53c6e2c559496700149560b3a94b94b8920e0705cd2487`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29b14ff43615dac430701017e1a95d84a605d40df7e69393e02bc78849368384 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-83f1d388f24fd87b67e44893cb8d2e4c2632b6696ded895cdeb97aa64d09c77f |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee20318a29ddc8a0440fa08803bc8b20d40b88aa8968a3f9568502e3d7840cef |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-213845367e0d63096dcfdf4cb616f1c3f7d99d59ac9ffbd1b5211f9d2db924c6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e07a66f5 | sha256-250c42289a8aea98ab928fe982bf8c36994af3ef6a3b4338c43ee21c36a17ea5 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e07a66f5 | sha256-198e091473a8aeddbd2b4f73ca0ac2287da8e85911935f45c5d9075373d3c4e9 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-c42881a8 | sha256-cc3f03b8c14cef0eecb7a10fb30230cf7050dc5db73b7e5be92bb94e56f27517 |
