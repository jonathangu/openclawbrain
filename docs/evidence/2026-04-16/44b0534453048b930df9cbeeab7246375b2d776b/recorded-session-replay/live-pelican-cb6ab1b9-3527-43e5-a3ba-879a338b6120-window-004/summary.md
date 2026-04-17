# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f94a31a75f6f674deb4ed72bb4e73c45b90c17561480d33d8d146b93540cfdaf`
- fixture hash: `sha256-55ad28e1e1c0e357b90d71c5a61455a338c2e0a4ef3a7f6c092d3616039ed272`
- score hash: `sha256-3a4b41d3c3342e3cad8e55f11b4435e87bc3f032c70d684158dd12abc9e88eb8`
- bundle hash: `sha256-32f967d8c775a30221a24bbe1f3f7231473ba0a580328bd8c938656111c480e7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a7e02cf5e88271092f868ed8daefe51bb787b99a9e0166c0444d9f0e9eabb76 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f028fd11258d3fbacef7a9e4c555ff66e556c7f811af5c7a0af004ac805867e7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3a858ffcf26a26d371fd4c6c80c6b5f94a675342fef09d07e6c9520e74057a34 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-916c8f6eaed57de015b9fb039a1bea0bed9e152182a2c920ee69862533db0a8b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-235d0f48 | sha256-5deae9a367bc0b42c0b4c4d7182da5f01f1e00cd4fa25b9c3bc70c6a573b2653 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-235d0f48 | sha256-bca45f186b46aace56fb91c33fdb160d3c9c971dadd1cf9fe4102d1c855d3d35 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-fefd7c0d | sha256-9d18f9f021d66338e118f6463c75e36baf6fbc84aa4d554af6dc8472449a24ae |
