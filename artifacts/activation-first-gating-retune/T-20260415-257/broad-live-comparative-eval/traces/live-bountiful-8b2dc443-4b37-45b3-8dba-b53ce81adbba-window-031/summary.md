# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c591f5c65a17f8b728581b0a64e58f54d550808c4d6d87e9681919456c4e7956`
- fixture hash: `sha256-7d10d338cf842d955d253c17c711f61a919941b05a2192e292201851e3214a2a`
- score hash: `sha256-5d43f311142bef640d60c974047bf03ca21bd49485eae90bc474883328828184`
- bundle hash: `sha256-fef2e810b1dbb33fa9d2e7e107288becf591f70f2c055e237412317697c2a404`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dc716d2fe029608c9fe52ecb8defed0c2de7ebf60cb8d8503f70a55a165b4d33 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c27eb0a2eae400315356e8e11d697f9add03897c50148fbc59b53569bd9dd6dc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8625a3bfacd347ccb4b251023c824e41324d860adb326f36be9257feae47f00 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7ae13bafb82d32bf4af54b7205b683866db0b1c947b09b86fc12383f7e59995c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-47c1687b | sha256-71d57d2cbc415478ce3741fa89cbd08c3d9c3f1f8b1f655036a591fb59b60670 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-47c1687b | sha256-a324d8afad6ea4079c6c77b1baadf4f839dca5c1210ddb858f7384a1436a5b13 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b116fb3a | sha256-a79bacdfc51d58ab2b8504d8326b073e29a2db6f7cbc00fdc85de6f0d4203cc8 |
