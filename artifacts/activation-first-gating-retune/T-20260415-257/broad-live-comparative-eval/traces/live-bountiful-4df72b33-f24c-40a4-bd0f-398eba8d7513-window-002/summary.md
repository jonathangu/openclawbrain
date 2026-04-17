# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-25e681dde9bf99a5066e3fd272c254e137908dd8248f9cd30c28377b5642eb80`
- fixture hash: `sha256-118dd0d43d47e09d3e0fb14557115fffb91ecc9b2c9362bf193950d5af577035`
- score hash: `sha256-2a0b18c59cc020cf5d8143754d87d0dff6233957f32a848ad43baa559420989c`
- bundle hash: `sha256-d12ad06e9a0c9a38ce0d27075a9b1ce6b9b861fe64b04e337afbbc0084def41b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | vector_only | 80 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 4/12
- phrase hit rate: 0.333333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-053f07f407b9f0886975eb3e4d95aa7c39bed9e8cf96e6716ec7a7f71273ccd2 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-72b41103d613a2d367de580b27f6fe879dd3d5857b79c3a6c7aa003635e6eeeb |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-a4ae8581eb81eda4dd51e6f55777c878ebff1761fdca4f7a19a55416681e0c7f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d9b8239b991b7a40d366cfaf2dc1d25f547a5ceec4abe0cb71647c135b972252 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-dbc1b8f7 | sha256-ad86c6dbec788ceb054b5d7c2ddce642b44b522c20962cd33524e85a4b710a61 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-dbc1b8f7 | sha256-ad86c6dbec788ceb054b5d7c2ddce642b44b522c20962cd33524e85a4b710a61 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-97ef3512 | sha256-fcffac974827ecd0fe877e387d66f479f6c6213bc022b9162d834f6ada3a05f4 |
