# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f4905564adc9cb953b8b5504309a4080c3ac583fe0f629cb62b1e05f91ea23a3`
- fixture hash: `sha256-0ad0b5e1e0f2271069ee0d118e38a8f083b22de4d11f9b10cb9ee63b3ed54883`
- score hash: `sha256-8ca4d8ada297eb1a6219c54398e2fa646be14fd8d3f88f836a54c17b8d740e08`
- bundle hash: `sha256-84a34d1235faf4dcb8667af55929436d0a9ce9040232222e7989d4f6b134059e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-64badee520388e2e251dcf80ba87d74776085beb63219f4be30791f06cfae40c |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-746f6b14569e0d37a2c038927bc23fcfe8777a31a7f62b443e2769ebed5fa836 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cd3538cb628bb5ccaff4ade045dbbc4e9f929c6ad9a0e46961c8f5e8c2860207 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-8bf35a2baba3006c73f03abf8c24caa3b48854d83b4a1633d4efeb55fb8bd9e3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8bd6b94c | sha256-138ca40ec4f20d8255b4cfd4ddcf385e283aaa9e39b3d12fc6ee9921d661fa74 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8bd6b94c | sha256-e5191054ab73fe0fe52b0681839d738d4675821446c4d0e287393e76c1d721b8 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-8bd6b94c | sha256-138ca40ec4f20d8255b4cfd4ddcf385e283aaa9e39b3d12fc6ee9921d661fa74 |
