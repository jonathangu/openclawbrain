# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025`
- winner mode: `graph_prior_only`
- trace hash: `sha256-255cb828cfc3b54a6910bd2ccddaffb0dd5a77237579bca151a6d4ac10b7f7ef`
- fixture hash: `sha256-6bc9cf75cae8800470997ef07fe6de902accade663aac8d00dbcb4e5cbc81f77`
- score hash: `sha256-c8be769772ca46843de185453d44ca25cf83afb3ede5a61ab89038e3d8d99557`
- bundle hash: `sha256-6ed6667bfaa644fdd985e25e0605b3aa126ed4c9af24c0c0fc226ec8f97eeb84`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-297b1309d9c33c67d42bc2f9d113a9c7958ac8f0888f1f62541fe7db873a4e3d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f3f790b99eb1c24ad5de5b970c264873a45afb428f3eaddcdba052ab9bb0eeb5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0e3324d509a1fc91e9674ac20490ed8c84e78e287522e4c3a015a746afe4c717 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4eb9e744cdab4f26854b683061c88d792da3464a89fbcee5b7b59e9df151dc87 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-53d0f817 | sha256-e6ae21badf9d3f6ee0ae4c76b0084f813e7e0775a008d93ecc9014ef0807d986 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-53d0f817 | sha256-f1dd7163e49ed068788a9ff6f75cc4ffe88c426a8d4bc468310a8904eb0d0157 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-1b3a05c4 | sha256-ba60d2df31c5567c06ecf0ae45d7badbcb8e538e34a1fb375b33a1c55ef1b1e5 |
