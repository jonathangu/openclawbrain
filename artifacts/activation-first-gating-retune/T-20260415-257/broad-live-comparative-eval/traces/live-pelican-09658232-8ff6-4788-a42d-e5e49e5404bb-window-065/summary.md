# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b834ef975b4127c9fa6cce4b12fc80a10ac45f8451c226003f814696763d3404`
- fixture hash: `sha256-2ef4285ae644d199abac210f4e94c99bfd3cbffd40a56868154cea15ccdb9a86`
- score hash: `sha256-3a637090bfe7acf2ef803aee732ae802b8005b4e85516d7f0801cf820c0ccc0b`
- bundle hash: `sha256-b12681371229e233b821ee8406065cad974f9ceb4e179ca2c2881c6f96a709b4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f63fdc694dc12e3fd1585f1a9d1a8d63286b83507ef0e36210c868d071e50d26 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2a6834bf4c312114d191ce52ca020d0b220c3f47cffba91af22ae26475b9e962 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6c0ed86c401f3857de85be9c0d80ef599a4be6d91acdfbf45ad75ceb835aaeb2 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6bc0b2e0d79f03f2aaed3807a9614f0b7564c302c41a5e49a7892716d9c26d08 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dd41d06c | sha256-825418b4a0efe2ebfa17f69d00eda2b6de11861b3477c15a4360a6ceae9c1d98 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dd41d06c | sha256-298930ef5dcfb5ad9bcbf442f602af9c77f08e20058a3a4c1b2669352c543a96 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3b5c2937 | sha256-3724b921ba28f0dfdf37cb23d8f12238d5a31007ce0ee69bfbbcb74e8db47ee4 |
