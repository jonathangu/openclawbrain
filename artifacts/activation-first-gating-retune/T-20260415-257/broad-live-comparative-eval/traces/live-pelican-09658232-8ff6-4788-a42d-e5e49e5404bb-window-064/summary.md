# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f39bc7349702eda18b0d056342226b1aabb41caee42927ea480ba26a62daf2f`
- fixture hash: `sha256-841eefc5eecc02fe972ed7cac8e3716da5b289fe7edcf8c461503d651db37931`
- score hash: `sha256-cfe3d57efe2d55e135bd057c1e4d3740079ad0c0acdc44eb2a4efaac24912602`
- bundle hash: `sha256-fe45e2bce561564443e2486b89a289846142155e8c1469dd3957ba431f5388f1`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f5559e04ec3c75dd16ee057dcaef2391dd2363ce8cb9ccfcfa727aea97487dcd |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-adca4882a9ce2a5475c054b8be6d914e9e08d4bbba905a5d614465369f63a9a3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a172156b540957c0c1a5f4d8c02d53d6f020a908c1c796cc072ece56c799ec99 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-9f46f3f14e1c64c4b17a7a50404f0e48498ddc8c9a034e46414cab11f38b9904 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-75b80874 | sha256-0c80c46b8a27bda3fe8bdf535065835742b88ba394b0aa8d604141562ba2ab21 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-75b80874 | sha256-e0f4b2b3e66d07b9d60616ea7a093a1aa26e9d7d1e9dcc4b806512f332a8375e |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-0ebad02f | sha256-a2d8a6c917cbac063400bc9c35e9cf5d36268b9a0166ef86d6c65430b1ec3fd1 |
