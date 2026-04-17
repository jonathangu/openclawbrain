# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b0963b5e18b40bd244437e7e0701feca11058e18371c7c7830706c53a28f15f0`
- fixture hash: `sha256-ad1fe96e9866fc7227d860e828f71679e46d996ff90526f19b5279748e32ad9b`
- score hash: `sha256-d79a2fd2fd313e108c1d23423076c51bf763a2944388501e46978d970fb8a086`
- bundle hash: `sha256-aa4951f8f7c641c59264d28f52d83604f39e0ebbd1530dcdfbb5f18477b15205`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-63768ebf632f25773108234ec4f8850307fca437412854c0aa69b01f97c1eac8 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-09032a8be01c062412ebc3e3f93cad8a55b29e496ba22ec34da1c09d69f6d5ec |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c0e224faee360916f1c27858fc0598408e5c8cdb3cf3257faf07c7b74a42f6b9 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-b6915fe9e5f0e00122c8d2a9f4657dd0b77205d2a3881a7d9cf12552a2d61182 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e43a5e1f | sha256-c58e5c336251511527bc47e0017cbcedc804d11b89c8a1bd3d18f0f151dfef64 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e43a5e1f | sha256-37912983ae76a49b654e4b04324480109f0f34a614c1a1fe0a9ebd142f87b9f4 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-9596a786 | sha256-94042c32ed2a5f5e0fa21c5f7bde393012734f29328dfb97434ba69ae9d5e9c8 |
