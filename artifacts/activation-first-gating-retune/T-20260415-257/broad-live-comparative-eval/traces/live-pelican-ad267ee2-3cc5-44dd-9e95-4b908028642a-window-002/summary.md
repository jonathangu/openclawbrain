# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1a576ab7fc82836d62896c5506ba892a7997f6c29eafb6387885075368088d2b`
- fixture hash: `sha256-e830bab1e1b5c601ab706b387c4f671be86f28c4ff56747b0f78265a86556170`
- score hash: `sha256-86abc7a7051117fbd301235d76d84add6c73c19bd428d2dbcca13ddd433483a0`
- bundle hash: `sha256-a8c24d0e34a785f34da8bb48cd3a9628c11475f729308c7c5840f12dadb49f94`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-309a76d5f65b7ffefd710af5c6f62a81606516631b55e10f450624750cad9788 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2b49744234430e1f48e0d0b79546c87167fbefbeab5a217e9b1f1a447dd505c9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9c0416489b979ffd9739d8d76b6d71a996697845db4dc25ca5813c2ebc5fb668 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-d55da1c5491577f74871233bb97894d8f5addda01707e206a1f6e62ed4fbeb50 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4ddb1eea | sha256-b1a98812782003344bb3663a2408a151a5d6b05cd54146b733baec87076e4bb4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4ddb1eea | sha256-151b3b739720892bc8982dc0e50ed1f388bdd1a74b026747d38db9f383d5a726 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-da9c824b | sha256-48b1b361271af57b6f10571bcdbe0a137a3eac762ea62ff24dfc017b5d421b21 |
