# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055`
- winner mode: `graph_prior_only`
- trace hash: `sha256-069c659e483d79099c9522902169a0e3c008a2a3a1a608f281e5842abe60c793`
- fixture hash: `sha256-2053a334b00cb8986b08e94b050daa206b7253e27c1f42496d3a7ffe4c19e5d6`
- score hash: `sha256-29bc838552c5ec8f0fc7f28ac016d6398b4dfbf3290b471494cb2d6a54eaec9c`
- bundle hash: `sha256-577bff3be358daa540c188d37d52bf984eb2f8bffda2ddcbe87034c91ad58025`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c7d0a9be6adcdff721e255d979e1bead77026cd16f5da5ab306eca424cee158d |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-826d2477701fc8dfca45d1c69732e757c9fb8b9265c512632e727c38195699b5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f26018956d8eeb37df45801dc0cda158449c31646510be01676b8a4cbed3d0e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-85b77fee3183d00b086aa56c8f8f82d29d342cc2f918b40255440de25ebea8f8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-90b56b34 | sha256-c36efd8a268a708bdfb7086c9fce8c762d5a9c7ced87ed49571df8f460906445 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-90b56b34 | sha256-892e72a7b4854605a2f4ac74b190c4d7303be9a2e95eb6eece295b1a87997ea5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f3c60b27 | sha256-1d98955e95440cb395e5530dfc73a1dac92c84b6d31adfce327a311206cabc37 |
