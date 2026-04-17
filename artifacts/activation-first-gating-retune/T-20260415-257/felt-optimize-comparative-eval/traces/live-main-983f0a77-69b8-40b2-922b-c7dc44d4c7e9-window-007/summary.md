# Recorded Session Replay Proof Bundle

- trace id: `live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13518408454d88b3ad692b956343d851ffe682724dcc9ea68679835cb38cd6f1`
- fixture hash: `sha256-d8ddfc141ca061b024a7735fc1bd6c41a09ad3c89f85b7541ee5a4463459f049`
- score hash: `sha256-e60ca388b13186a1428d6825d13a13790e82ff2404afb3c31e896245ac4e44e3`
- bundle hash: `sha256-db5f03278bcd7c0a673cd7cdddfc0e40c31ac1353b8f2aff42d4fe87f69dbc47`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/12
- phrase hit rate: 0.083333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-76a9870f23308038c7dfa2834df546254ae4769b20da16b32ac7e7ef5f9b078e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7a5c7e1b3ec169ef7a375a2aea3263a1dac5142b41c2e0e6e608d917ed33f601 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-f312017158c7771635c600d07d776d222bb2450bca7034ceba1e0b989c444dff |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e2778e2f69254508110d4814922ba1835ecb352cf4220d0295bca747e3c7c956 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8f470d63 | sha256-8fde858e37ad82f2abca1e2f45977d1347c517b1fa500eaeb039756b61bf31f2 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-8f470d63 | sha256-89b4f89215afb7cb4ccb0e608bdb9a5c6af4cb3ff6ede5d5be6f2fc0958e645d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c0d72692 | sha256-e1251581deb23454237dd987807d8821ff539ac821ada43d76ec8244755836c8 |
