# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f69683abc74146be49e8afbd73d2f629322351b8f1ff326bedad7089f23b35cc`
- fixture hash: `sha256-78ae89352ee0e2620fdc9e4b5d6b74ee70bb4cf28775ccac9315ef7f4b6b2525`
- score hash: `sha256-554e914d2a86af967e7ed5df7fc747114d85dc658c1b56f27266edd2c2035333`
- bundle hash: `sha256-b86f1b4d05ecd83ec59ec6f0cd7e4a3673b53f111b0efc9714fd111ceca5597d`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-81a98d75515ca1c6519d32d4f8b5120338f9765022c93b90e0504e9561ef38af |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2c2b29bc492d768ab34b4ed413a25f117c7392aed00c556e4744b159c58e1083 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cbd195b4528dbb454f6acba3bba570f6acbef39e94c76f000831311aa62ea852 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f50c9727d6b569c6032f36a62d197739ab8af3d7a53d937d37e220c284e2f9d5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f3918232 | sha256-3da624721fb9ba49dd7bd9d967e6473c1891229909a36334f04d015122d6e31e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f3918232 | sha256-ff29166add6d3ac5544451e325d6d29a28cff2fa131067a2844c8ee92e24df68 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-29e4b81f | sha256-264e7d227703983652e1c595a8c26719043d941217b87931f74ace728f8590ed |
