# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1eeac1e61d3b2146227988d62e1fe6c84e0b77c1468137fa8c6d382736c2c4ff`
- fixture hash: `sha256-1569517b028e54a6250341eadd5d277f396164c98c963a62234840e80af05420`
- score hash: `sha256-24c8e5d95cd5b6a5e9d6d5d2d9dde3a714a62a476a7d69b880c2e7e96cca3615`
- bundle hash: `sha256-8938119e7eb1d8cf2b035c9ada81e51cd4f2ca79a8c73ab7083254fed1f4b126`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-1ab43396288e04b0918dd42961174d9ff98e67ef546a7920bded838cbb02fd14 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-019516011733590437990f5518268f5602cd4936115d2afb2a9933b07b865395 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-f6970631a2378fed2cd42d037aa498f356e2f0efe9c0f97625c178b5e1c577b6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3c9ad9d3 | sha256-9bb18b72d3961994d64b1db7720c1b495e33dcf3324ce72f01859d9c7c8d072a |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3c9ad9d3 | sha256-919de36380b01cd41982fa0f4669c9ee7355d159e1e78ed5f2fab0e1975e0853 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-3c9ad9d3 | sha256-9bb18b72d3961994d64b1db7720c1b495e33dcf3324ce72f01859d9c7c8d072a |
