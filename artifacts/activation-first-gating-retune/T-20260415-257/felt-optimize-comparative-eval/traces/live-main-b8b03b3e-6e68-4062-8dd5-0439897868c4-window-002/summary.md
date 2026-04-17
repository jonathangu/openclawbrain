# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68b2fa5dad518df8f3866150e5eb5ae2df4d40d0d1730a7b39326babb425756a`
- fixture hash: `sha256-9c558cf390c2d5519271f6ba91a97c5aab0727de8cfbeaa1362c2e39d2a00c20`
- score hash: `sha256-674ad6707ac1d052c05a0c79440b0d7b7f78c6734e4b1be5356fc16845b7aaab`
- bundle hash: `sha256-044d6bf7d84b68e4649354e5f89361f0d3940c8c4d6c704a8f58a92ff296113a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8709c6589a780862225e4afa90cbbf44ed4ef4f7b39772bdc54c0a9f8a33087 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8eaaace837dd4dcf098eb5a5bf86e8d3b6c0fb54a587f17a893f4e5fed08d993 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9c6476bd5e106fc1a91eeeff254952c27a9c2f3cc8088e41da99b34ec78cdd9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9571a2f88ad4cd1014db1b982ce5394119398af6eac95fcc69b5fb1e1a5b2312 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-63ed0dd2 | sha256-6976fb7653e70816ac9397effe3a0194f0f44c2ddc564f918d32d959558b2717 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-63ed0dd2 | sha256-dcdf0716af4e669d6aa75716919cec84eee241934767ee26cf1e40b5f4d27cfb |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d0aae3bf | sha256-30ec8ae5e7dca56b4da3d4bb80722750dc4476f1a72cf061182b2a40478691fc |
