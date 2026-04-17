# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f39bc7349702eda18b0d056342226b1aabb41caee42927ea480ba26a62daf2f`
- fixture hash: `sha256-841eefc5eecc02fe972ed7cac8e3716da5b289fe7edcf8c461503d651db37931`
- score hash: `sha256-53a880b9b9342555e8400250adfea09183a2ded669c623e5e4e42b5e2782cc1c`
- bundle hash: `sha256-6b208c32538da74f88ef153abfba9b64ffafb396ff87a79d57e7299921f1d7b5`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f5559e04ec3c75dd16ee057dcaef2391dd2363ce8cb9ccfcfa727aea97487dcd |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c0f2ac9b4b48bc01d5c17b2349ceaea2e52698ca3e3bb267941ad57e5a2b3aa6 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1e7c2742a6e7cdcf81ca6d4260fc18afd5a140c0bc075defeaac90a81098a8a2 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f9a5c66593dcca287091fbfbc19280c6d7f98ec6df1d1b369b6f381896ee9dcf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4c98e86b | sha256-3eaf419957f2fc48c319de7dd73db08d0bf83b831b8b749a70005032a4c1cb8b |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4c98e86b | sha256-6e949a784a390581942500378ca9ef0205593fed4dbd52cbbd6fa54cc89e8135 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-e59bb026 | sha256-ca5bc644d74d0b39de4d72525d3a4f683fa9a5c712984f99d1a5a6e959b0ff64 |
