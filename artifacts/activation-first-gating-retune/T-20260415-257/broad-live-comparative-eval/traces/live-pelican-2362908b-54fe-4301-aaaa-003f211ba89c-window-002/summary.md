# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df541968ca52654e5efa48a1a6713bb4511f8366d389ef30e36174b0478a0f72`
- fixture hash: `sha256-10a1d9d424d59bf74d6edef2d25c3d9864b38e04e75b6ff4b28dfed92245cd1e`
- score hash: `sha256-90febdb47b3a59e04ebbaef2cfaceda377025f2644ebb9013c74b39f2c994af4`
- bundle hash: `sha256-cb7e13b59660dd9e80c8e9d169b51ec0089a25ecbc86ef999afdadbd7e5b3731`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe456f8f9f99c14a6c26ae3cfe1240fa644752e272eded6c0df3fca37912d301 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aa60618b94f564b00722cf219a6caff0e3b1d3ffdd03c2cb5f123e4b682554aa |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4c81219b41929cef7d8317aa884b2eedff94d25ccbaba1f149048b74eed1b365 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-0c3d9aa80225ff38ac160fa21c9118dc44a649827e53c5e4f07cbb12a0bec457 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-05a07367 | sha256-cd8c34e6178f9e951b962026dd567a65d7e4a507f200457309f1bbae017e17c8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-05a07367 | sha256-2ce092bb463bba87c592a22a2dfcc5fd391bf4db1532fb6fcd3b785de60de985 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ad1f5620 | sha256-19b1cb05a01f4dbb5af9a6ce70c896c6df52beec4b6826e9e45133f81ddc319c |
