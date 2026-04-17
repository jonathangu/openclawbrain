# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f07ed34ccc6a5c54819d12a1e93195c70560e32cc80e0d0e09592b4765b8105`
- fixture hash: `sha256-00d9f388b90351cc79a6666fb1faf09e6f2109bf7c85e8cdc18048263ccb39a6`
- score hash: `sha256-2011190ed8570bd22f5f6fa0c006bd212bf2f6453fe56d34cab705820fd80156`
- bundle hash: `sha256-c5d5412a525e3185786a800bd911c0a5727fbfe54ff1970fea5bd1c1a7398d6e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f9e400fcbe43d9ab55b6048a20689714c3c7aae22f85e1babf49f3474335a32 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d891270ab984c9de067b354f904b23cbddd392a5c0159e62cb0ee48cc2198438 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cf1c0540ee821679b3ccdf68feec6e157849bbb46b71ebe66f924b08936388aa |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-8f3280a485fe3c8b8c8bd14e9cbcbc25ad29e40b1de0fa8607ac26cd9f42cf35 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-70dfbbd5 | sha256-73beb51b7345f5b85319932629caa8e008dddbfbab411719a5498efc8bcfcb85 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-70dfbbd5 | sha256-9e57f89e7b7252d13e0cb4a7a9b5bee3c0d68fe9876ef4a286454e4ae04ea420 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-dcbecf78 | sha256-dab68cbb02ca8aeef9e77a1e03b9d97b98e221eb1cb2b4bc87bdf76d98c77f18 |
