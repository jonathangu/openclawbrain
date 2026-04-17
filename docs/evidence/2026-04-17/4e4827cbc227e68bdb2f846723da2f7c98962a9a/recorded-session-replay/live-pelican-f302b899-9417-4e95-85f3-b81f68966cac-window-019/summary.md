# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a95260c17a69374ef7a9ff20490cb415b09868b4babdf035ba541b6d82beb5bf`
- fixture hash: `sha256-ff74599acd0d3d5ad2046fb7795a787fe8fa0e70837c98ae65f89838fc9f50e9`
- score hash: `sha256-9631db8e6e56d0bb4d8cd6e15eb05a66349b3f88dfbff0cefc1fc0922d903f44`
- bundle hash: `sha256-722f1d3f1afb16b6dd8f53c4d3c3c27b36558688e793fdb3ad2fb3e9f454e2dd`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30f22d5abba84d169a9ef0f72b28eb7bd4c2afa26a7910c928c371f416decf04 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b6c34a365445e3fea2e7c29d62bc99990b96cf08ea90a46346818bb362d2f281 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a53568de44c7aeea9fc4ee99e7866c585f9fff5889d6a4b5f3ccb54d03953c1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c16a9be3a5043d3c4f2f8ba0257619809c7de0ad1cfbc9cc05e37e39f68047aa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e1ffb3d4 | sha256-16ad2b4ee97d43a8ce0f6b02de12a1966510e9a43c6b3768567547fdf9c4a363 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e1ffb3d4 | sha256-b6ab242cfe73ff48ea7d0969036c7a214df38659d4df323191bfe07dc55bdd2c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ded4c18d | sha256-1a10dd4009930314012afd62abb8a8a99c03e4bbeb96707b586d5dfa0df49fdc |
