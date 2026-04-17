# Recorded Session Replay Proof Bundle

- trace id: `live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a7dfee8569cbeebae47062b33f26bd559b44ab7e32ac7a65f3d53fcd4f9d6446`
- fixture hash: `sha256-bf2c49e43d0148934d94e443780f19f84be1befb9f46554500ee32090d69fd0f`
- score hash: `sha256-8ece7580aa3ad631481cc406c5681c14378e846f48cab7e83efc936d1538ba4a`
- bundle hash: `sha256-465f0a0d1a2f1ca5d52c886e1a9e0cf4c33cd7a723aacac1f67cadffff733670`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-aefc644f475f6e64faeecc10e1bad33424cc557b74533b3b9b16e76adc362925 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cfe14bea8fa607474eb1f9bcf863186c28819c8fb57b7df7da5ea83cca4251c3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-037ae7a9bdd6d2901a50d5ac9e5e60eb6b27c5074fb33cd83151926449bb27f8 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-461112042bd7639b5459317e54c3fec187376680cee7b7741c48b329e9833ec3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b02ef600 | sha256-3f936c12e503b62ec2e59207de98b6dc8723c2b97b9d1fcc58aa2bd39b7b5bb9 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b02ef600 | sha256-71f9e17ce859606a1c96c5cb91528159a7f431c579edcab760d907b72b0788f6 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-02e34eb1 | sha256-9d1447cfcf7d7768f12e8a1a67b0ccf3e2832846e13cb039bf7685634124e5ed |
