# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-529373cf8f7314054ee5a9938b5133a303e70a9153c03b373cae4ff852f394c7`
- fixture hash: `sha256-c16690eb3752325552dd8dd957f6a57c852c3d697d1ce7463c9556556d92ca19`
- score hash: `sha256-de19ab130a874e03881a4db853083d81458e33b5f7f09e9384e5414e7efc22a9`
- bundle hash: `sha256-60a1e65e373d564ea2fe1e237de18c3764a619b60d77f1a160c4c410b12d8407`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9641da18215ca2d07fc313a19aa471e30d85d3a5754d470ceff969f5080d786d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-822e422bccc81197dfd6c9c69ee1c5dbe119e1c55ae701d89b6dbc6dc3256583 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2ff0032df102871853e00e771b758e8ec1ff37faa6e4da14bcc55c3127169c6d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2e9e0fee32243d43dfd28670d4339ce8eab1d9ce03a8f04bf79fbaea8b6cdd5d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2617e699 | sha256-be53eb8a4b1ee2799d34fadf15396f49dff6657dd40902fb46d8dfcf7d12a23e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2617e699 | sha256-1b2fe13593bb4e3b8a2c14a170d5be3054a162bff98ffeb91aefdc296de1df45 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d7d77704 | sha256-72dc976874e6f35f7a7815b61ba6efbefe6451a231be077ff6be818a9f4ff535 |
