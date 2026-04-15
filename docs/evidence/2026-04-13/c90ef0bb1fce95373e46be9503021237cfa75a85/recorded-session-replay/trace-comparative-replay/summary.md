# Recorded Session Replay Proof Bundle

- trace id: `trace-comparative-replay`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b95f895c64041d1808d9fe91f67bd8bd003d088a62ca59cf1e440938a201e26f`
- fixture hash: `sha256-279a2c9838f639bc9a2c4c0580126d227be88558a0c39fd66ed2315cce401582`
- score hash: `sha256-97413340d5b8b24429991c8cea4e5dc72cdf1cda321f80aa4d43151d07a555ae`
- bundle hash: `sha256-3e58e5403b322ba24ea7c4e9b2d1082255c388b5f926c0537508a8d877d290d0`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 6/8
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/2 | 0 | 0 | 2 | 2 | 0 | sha256-bf3e5467de37132a2e81d2ad5eb47a8c3fbe8fdd240efb52eab44760e5cd2955 |
| vector_only | 2 | 2 | 2/2 | 0 | 0 | 2 | 2 | 0 | sha256-f3fd840346066a3b80c4d9368196d80f2579c22464796384ffd347e86421f85b |
| graph_prior_only | 2 | 2 | 2/2 | 0 | 0 | 2 | 2 | 0 | sha256-d275e4692c645965204c3e5211942c85ca0d5ff0c75382c537f3ce1f501ed873 |
| learned_route | 2 | 2 | 2/2 | 1 | 1 | 2 | 2 | 0 | sha256-6d9c8c21ec3ccbdabd2a4812a7f67419b68a5d464ef4d2323e76e62178765ba5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-2 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-5d048fb2 | sha256-1059460d130da3ed0ff23cb4dc1327d7c26d5017ddc765ffe88f5a2979bf087d |
| vector_only | turn-2 | 100 | yes | 1/1 | no | no | pack-5d048fb2 | sha256-1059460d130da3ed0ff23cb4dc1327d7c26d5017ddc765ffe88f5a2979bf087d |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-5d048fb2 | sha256-1059460d130da3ed0ff23cb4dc1327d7c26d5017ddc765ffe88f5a2979bf087d |
| graph_prior_only | turn-2 | 100 | yes | 1/1 | no | no | pack-5d048fb2 | sha256-1059460d130da3ed0ff23cb4dc1327d7c26d5017ddc765ffe88f5a2979bf087d |
| learned_route | turn-1 | 100 | yes | 1/1 | no | yes | pack-5d048fb2 | sha256-1059460d130da3ed0ff23cb4dc1327d7c26d5017ddc765ffe88f5a2979bf087d |
| learned_route | turn-2 | 100 | yes | 1/1 | yes | no | pack-3d31386d | sha256-bccb4f2d723a77d6bde9203fa1c227642bf15bded08b508365f4ebaa354fbc1e |
