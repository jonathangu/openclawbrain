# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049`
- winner mode: `graph_prior_only`
- trace hash: `sha256-10f30fda1583220ffcb0e13cb73de4976d5f3f5f0f058e8e816ab9eaaeb4bc0c`
- fixture hash: `sha256-81aecda5857d0ab09faf0a56bf49fbe289e64582b0578df3f1535d5bf05ea11e`
- score hash: `sha256-b683d0a2c0a705da10107d5a473ab49abf6d57cc23de9b1e2fbc9c49a490afbe`
- bundle hash: `sha256-f439417c47d5a2117cd437fa12a4397007b1e3b95883156a7eca2da7dc058cf8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd49736decd703dddc6036cdf0bf744059f6270cb8728fb209a65a281dd21058 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-16add9eff871eea473d28f71fb76aa36b63cb0e198ceb16e51b6d97e1aa04fbe |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5ccadac20cdda48a7a802b2c224847a85a8e5a15bf46beef88940f0ccad64b2 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-be1c6402583def4e0c4d3e53170bf53bd8873510b82e2c2c5bbbdbb918355b6b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-460ac314 | sha256-920327df44175e989a5ef7138549ff1af73e00d102bf910d17ea587c112f58dc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-460ac314 | sha256-f10cdfa7fe1a35247967c1b2f8ec6bbd9f0bb3ae563ea5b7aa7d7b122e039754 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a434978f | sha256-a13f54f9143b9fe3f1b6cbbdd5af4f4871437f91846a23997bf5d031f7c97ced |
