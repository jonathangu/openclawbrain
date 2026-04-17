# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f07ed34ccc6a5c54819d12a1e93195c70560e32cc80e0d0e09592b4765b8105`
- fixture hash: `sha256-00d9f388b90351cc79a6666fb1faf09e6f2109bf7c85e8cdc18048263ccb39a6`
- score hash: `sha256-2f8be160e9a6d971f7c200a339f9c6f3e3f10ebaa8637bd6600d7a22e28adeec`
- bundle hash: `sha256-64c76dfe0e4da0da10006103552bbe2b2dcce11053874d130fca5fce4ecde477`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c444b171cac98cb3d8cbe1b0651ee43812f1d9fe782cf67328569c3f24f26918 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-594089bc91172671dd6ff0868c324af4ee0ad5e47aa00ebbb4907c85863a2586 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f39caf4d5208423228d3f858e4fab418f3eab636085c8c7cb4821b8a0b901f73 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-41f7fb60 | sha256-4f4430d60d3694fd13faef2bfccbebc5108332f2fbcdf36adce7af9f642b45ea |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-41f7fb60 | sha256-027d16d88d3c2b2fe28e06ed23fcb759c082cc06b944e59454bc373778c87e3d |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-add70f03 | sha256-9af9da6fc3aac7220c673f882b08256a51b57149ada866a1e922c35e6e2044fe |
