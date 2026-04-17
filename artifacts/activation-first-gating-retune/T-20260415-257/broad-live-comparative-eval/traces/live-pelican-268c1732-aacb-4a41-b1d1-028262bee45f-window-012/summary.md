# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d0b5d294f5bcac07c81e1e9b7fbd08fc02ac60b4a0afd2bd2ad3564216748c02`
- fixture hash: `sha256-10c2797f7098132dfd19e74471fe861e4fd990acaf92ba667dc395a281a0c32a`
- score hash: `sha256-6de45d24916c5aa03b580ebf6fb5c02fc487ad0421da2280ad43edd61c9e4675`
- bundle hash: `sha256-2520b25ef85b347fa1e5cdba2a90a54bd52cf2bac2b042bc39b161836b6a8d2f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eb29fb5cb8fb01ea6e12d04715c0ac66ad31c35de2501ab2ab9a23569a1d387a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cf1b5fd7185d7e4e8a009bccac387ef0eeba04715e0e531fe3f1450fed9173d4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7fe024b222cb1e7ada031695cad625ab7e855217e26b475a3cfaeda484f79305 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-bb54e8e5b8f131fea39cdccbf55d81481f15ae098148f56f45a46733c142094d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bcdc5d27 | sha256-bf2edb082e218d822c199bdc600a2a0faf80911efcfaed112e7be4f135d1d9d7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bcdc5d27 | sha256-6e7907725313602eed4c66dedd442fe56d77034c8fa95abf6588620b6b3da1b4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-12195710 | sha256-27ec1b4e1228ce52b14b37c32bdd907cd85d0b08679a808cf51067cea7975c90 |
