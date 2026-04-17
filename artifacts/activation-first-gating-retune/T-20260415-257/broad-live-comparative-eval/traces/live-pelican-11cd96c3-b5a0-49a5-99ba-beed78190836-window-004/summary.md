# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7ce9d77f8d5b34f5d4a2ff238035837b9a17936c8718ddcd44e0135af5ed67b2`
- fixture hash: `sha256-b278e1b6b555771ff403bacda1c9f56aa4593110af14f3b45502af98316b55cf`
- score hash: `sha256-2943f5ccbed518d5be4f90abeabe57c5df72e36a09df7a2d169859864206084d`
- bundle hash: `sha256-3727d6c87e0f036b9c956efc71fe077acfd92cfc8c825cecff1ea4b0ef4c2ec1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1fde92dc9f75c3ed7b5cdbc92af57a8fdea90f988cee9df5a6592eb109fc517c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0aa4742fc31f44f7d0b30aa2759f9619059dbd4bad6349f6510c7b689f8a9e7d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b88492ad16f1a6d1ae2f04c810826409fc255c69fe4d6923d3736cd53163b0de |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f3b69d62676643b69ae32a0adbd9a75b08326dbb9a053b76974a2fa5277152d9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8299515b | sha256-1958527a3b9adfc7ff107ad8f24217a2668efe5e0292e932f4681561ff0dd283 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8299515b | sha256-e925f7bb1164fe1da395f596b572d75b83cae0edd32ba06b4dc4658afa08e66d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-498f9c28 | sha256-41eba997f09d87e6abd8404cb68b21b4a7acc9320e4b106a604233ccdf227ce5 |
