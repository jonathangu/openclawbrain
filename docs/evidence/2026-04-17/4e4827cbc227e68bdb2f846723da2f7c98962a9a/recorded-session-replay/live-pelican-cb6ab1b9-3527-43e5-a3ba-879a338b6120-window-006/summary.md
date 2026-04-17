# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f07ed34ccc6a5c54819d12a1e93195c70560e32cc80e0d0e09592b4765b8105`
- fixture hash: `sha256-00d9f388b90351cc79a6666fb1faf09e6f2109bf7c85e8cdc18048263ccb39a6`
- score hash: `sha256-f9dae07a13e7013df0371bc5f7b4902abc82c1754d362e4276bf2a5fa941dffc`
- bundle hash: `sha256-ca73e330925da51e93b6325bdd792c0d81103fe263b656b1000fddd40cdfb71f`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f9e400fcbe43d9ab55b6048a20689714c3c7aae22f85e1babf49f3474335a32 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-77f8b0f4f6f1475789067c2f7680641735c5f5a51d2dcb01d531a1811368377f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a9327cc2d6d48c1da0c8a6164b8cbadee5ee8a568fbc92cfb2e655d8be72014f |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-6adc9c9874063bf2cd3a7a92f48471b045b8c1ffa2249f2a597cf616dc985dcd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-09b95d77 | sha256-7214347baa26ebc7d4f5ee7782186f4a17b111b9a015a3196a7b9bcef2209c86 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-09b95d77 | sha256-a9c3baa6e8cc1d0b8a90814d6eded42c2505fc94702ba0791f5ab257558ed52b |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-7598711a | sha256-5bd0eddd74b61578b0cded1ddd5acf20bfaa37e2ce2dc7700911584446ee122f |
