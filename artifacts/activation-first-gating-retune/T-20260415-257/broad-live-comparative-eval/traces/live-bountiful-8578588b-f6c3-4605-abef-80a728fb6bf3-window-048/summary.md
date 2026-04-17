# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ec11702cd5c290669dcb9e538d05099a378cb83776c59f4106b510b78248b8f6`
- fixture hash: `sha256-c6a51fb2365b547d2ebf3546bcc8b17b6e2bc89686df2016f759916179857243`
- score hash: `sha256-ef2db2cd9a873894f1f769a58fd4f5d1eff0d3e67130985827c66f99c609da05`
- bundle hash: `sha256-32586eb8c43835570dfd014046c1f55373abc9fd021bc7793d350fa8f6cceb08`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-733cbd9351d42e7f14c5829edc01a658388a245c6f3afae8ff24b34e8acd9e00 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a328a165a82c0e3111c45dc62e6a0d93f45fc334c57e537f612113c5be78fae6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-30f1d30f2fcc62872b257f8b9f0e88bc1c679395178598479c6bb3bc0f078950 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b47c5f0c4b66abd2744970a47e6fe46443a641d12e66056005c7e985a309aff1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2b03b687 | sha256-59b7eb2fa4512bd72767c56ef86c0ced177e4e73b61d720c117b8a3e3c86c36a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2b03b687 | sha256-1f279fcfa1892271c0d726131c212bf8694bc1fa38c3d4d37705fbf07035ad4f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3ebdd01c | sha256-c6b121199240db3d0caf52c886a8561135956f6f7e1a93e4a1d16fa5c58208b3 |
