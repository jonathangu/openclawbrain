# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ed82473cf8a44c6f378cb688937e33b3ea351a6801142726a4915ee5fb6d88a`
- fixture hash: `sha256-4909ff1896b085966400449aca0e9ac319b4cd9d22c11198c9e8e1d61fedcf2c`
- score hash: `sha256-2a1cb78c66e1da98a465300cfc8a7c3deac4abdd81f82af5794e03874e7c3ce3`
- bundle hash: `sha256-b6da0e3c364e0a15a0d9eb03a81d60a598c6863054f410d949b06f57989278bc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46b89f268ea78eb5e49f9755003f2aa744b81e1b854e2ac1c9e8f1a95cc59955 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-608ba6c17c003274f4b9e55dfd3ced558f81319f9d8b85bcd689b3aef603bfdf |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c6a8046db8cb56f6ef0d82320c17cc2bfcb454c0621d804c3b9b074fc91fc53e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-4b26cdf91d8bcd620dc64b0456e4e4872fb5f30f867832dcb218250fa3a144ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bb2430b4 | sha256-9406ccba061c4517f8bfec9f372ae998bddcb7091f660fbf18ed7489d0a6dd29 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bb2430b4 | sha256-3d4f1a2514d497b98e5c87badaf4d543a1d1c28557e05c8d57279aaf9ea39bf7 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-bb2430b4 | sha256-9406ccba061c4517f8bfec9f372ae998bddcb7091f660fbf18ed7489d0a6dd29 |
