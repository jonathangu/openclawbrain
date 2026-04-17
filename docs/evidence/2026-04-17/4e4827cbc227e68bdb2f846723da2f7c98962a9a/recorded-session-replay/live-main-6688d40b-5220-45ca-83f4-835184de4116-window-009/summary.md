# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009`
- winner mode: `vector_only`
- trace hash: `sha256-1eeac1e61d3b2146227988d62e1fe6c84e0b77c1468137fa8c6d382736c2c4ff`
- fixture hash: `sha256-1569517b028e54a6250341eadd5d277f396164c98c963a62234840e80af05420`
- score hash: `sha256-70b9287fe55dca350d6bdcee81ecb09f248288755e5513c94537bd1f12084ac3`
- bundle hash: `sha256-102e8e64924a85e0f4deb6a6be2e8503b9ebb99cd54b697159d84262f74b9ff1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 60 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/12
- phrase hit rate: 0.083333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5439b064337936d23e3ce0669d08085a2f0dcec2b235478161f6d9e74cb033a |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b228ba62372845aabce0f9d655d138a67be1ed6cfa746d42bacb4fafdc07a7ff |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cdb67074c87ef8317da6c940111a158f55ad3d0d2f9534e589c3042528b89cb0 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-85e58a895538184232609d27122ef940d55ec0e5baebda0653b6f545bc9aed47 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-2f4880ce | sha256-79960ad7b32757fb28dbf068cb64eb9b38509f5c89d4cef5735d314b55aa96f4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2f4880ce | sha256-f50120826599a74af3d21e6f0858804dd42a4d3f729c3ea3b49db5f7b68dd43f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a1e3b9bb | sha256-59945829a27f194a1645b83c8c401ed5ab2a4e9f9b7229df9017ead3d6dad6df |
